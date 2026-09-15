import torch
import torch.nn.functional as F


def get_discounted_returns(r, v_next, discount):
    r = r + (discount * v_next)
    return r

def get_actor_loss(
    actions_1,
    advantage,
    log_p0,
    log_p1,
):
    """
    Warning: Likely contains a mathematical error which weakens policy gradient!
    See with_logits version below which uses the log_p of the *combination* of 
    actual actions_1 rather than each bit.
    
    :param actions_1: Description
    :param advantage: Description
    :param log_p0: Description
    :param log_p1: Description
    """
    # Compute log probabilities of the selected actions
    actions_0 = (1 - actions_1)
    log_p_selected = (
        actions_1 * log_p1 + 
        actions_0 * log_p0
    )
    
    # Weight by the advantages
    # When advantage is masked to zero, weighted_log_p will be zero too - ie no loss

    # Advantage has shape [b] so make it [b,1] to broadcast in action dim
    weighted_log_p = log_p_selected * advantage.unsqueeze(1)

    # Compute the policy loss (negative of the weighted sum of log probabilities)
    policy_loss = -weighted_log_p.mean()
    return policy_loss

def get_policy_metrics(
    actions_1,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
    logits,        # Raw model output [B, A]
):
    # 1. Single pass for log-probabilities
    log_p1 = F.logsigmoid(logits)    # log(p)
    log_p0 = F.logsigmoid(-logits)   # log(1-p)
    
    # 2. Joint Log-Probability of the EXECUTED action
    # We select based on what actually happened
    log_p_bits = (actions_1 * log_p1) + ((1 - actions_1) * log_p0)
    log_p_joint = log_p_bits.sum(dim=-1)
    
    # 3. Policy Entropy (analytical, from logits)
    # Use exp() to get the raw probabilities from the logs
    p1 = torch.exp(log_p1)
    p0 = torch.exp(log_p0)
    
    # Entropy per bit: -(p1*log_p1 + p0*log_p0)
    entropy_bits = -(p1 * log_p1 + p0 * log_p0)
    
    # Total entropy for the joint distribution is the sum of bit entropies
    #total_entropy = entropy_bits.sum(dim=-1)
    total_entropy = entropy_bits.mean(dim=-1)  # I think I chose mean for invariance to batch size
    return log_p_joint, total_entropy

def get_actor_loss_with_logits(
    actions_1,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
    advantage,     # Advantage [B]
    logits,        # Raw model output [B, A]
):
    # 1. Correctly compute log probabilities for both possible states
    log_p1 = F.logsigmoid(logits)   # log(P(bit=1))
    log_p0 = F.logsigmoid(-logits)  # log(P(bit=0))
    
    # 2. Pick the log_prob that matches what actually happened (actions_1)
    # If action was 1, take log_p1. If 0, take log_p0.
    log_p_bits = (actions_1 * log_p1) + ((1 - actions_1) * log_p0)
    
    # 3. Sum across actions to get the Joint Log Probability [B]
    log_p_joint = log_p_bits.sum(dim=-1)
    
    # 4. Weight by advantage and take the negative mean for the batch
    policy_loss = -(log_p_joint * advantage).mean()
    
    return policy_loss

def get_actor_loss_with_joint_log_p(
    advantage,     # Advantage [B]
    log_p_joint, # Joint Log Probability [B]
):
    # 4. Weight by advantage and take the negative mean for the batch
    policy_loss = -(log_p_joint * advantage).mean()    
    return policy_loss

def get_critic_loss(advantage):
    """
    Critic loss is simply the squared advantage, i.e. value function should converge to observed returns.
    """
    loss = advantage.pow(2).mean()
    return loss

def get_advantage(
        v1:torch.Tensor,
        v2:torch.Tensor,
        r:torch.Tensor,
        discount:float,
):
    returns = get_discounted_returns(
        r = r,
        v_next = v2,
        discount = discount,
    )  # returns shape [b]; r+d*v_next
    advantage = returns - v1
    return advantage
    
def normalize_advantage(advantage, epsilon:float = 1e-1, clamp:float|None = None):
    """
    `advantage` should be a 1d tensor of advantages over a batch or an episode.
    Normalize the advantage values by subtracting the mean and dividing by the standard deviation.
    Note this is typically only done for the Actor.
    
    Parameters:
    - advantage (torch.Tensor): A tensor of advantage values over an episode.
    
    Returns:
    - torch.Tensor: The normalized advantage values.
    """
    # Compute the mean and standard deviation of the advantage values
    mean_adv = torch.mean(advantage)
    std_adv = torch.std(advantage)

    # Normalize the advantages
    normalized_advantage = (advantage - mean_adv) / (std_adv + epsilon)  # Add epsilon to avoid division by zero

    if clamp is not None:
        #normalized_advantage = torch.clamp(normalized_advantage, -clamp, clamp)
        normalized_advantage = clamp * torch.tanh(normalized_advantage / clamp)

    return normalized_advantage

def get_epsilon_greedy(batch_size:int, p_random, device = None):
    """
    Implements the epsilon-greedy decision for a minibatch, 
    returning a tensor where True means do a random action
    """
    r = torch.rand(batch_size, device=device)
    random_indices = r < p_random  # [batch_size] bool; eps-greedy is P(random action)
    return random_indices

def get_random_actions(batch_size:int, possible_actions:torch.Tensor):
    """
    Returns a tensor containing copies of random action vectors from possible_actions
    for each sample in batch_size.
    """
    num_action_combinations = possible_actions.shape[0]
    random_indices = torch.randint(high=num_action_combinations, size=(batch_size,))  # [B,]
    actions = possible_actions[random_indices]  # B, A
    return actions

def create_p_dict(p1, eps=1e-8):
    # Compute the log probabilities for action value = 1
    # Use the sigmoid function to compute probabilities and then take the log
    p0 = 1.0 - p1
    log_p1 = torch.log(p1 + eps)  # Log probabilities tensor, shape [b, N]
    log_p0 = torch.log(p0 + eps) # probs, log probs for action value = 0
    p_dict = {
        "p1": p1,
        "p0": p0,
        "log_p1": log_p1,
        "log_p0": log_p0,
    }
    return p_dict

def get_entropy_with_p_dict(p_dict:dict):
    entropy = get_entropy(
        p_dict["p0"], 
        p_dict["log_p0"],
        p_dict["p1"],
        p_dict["log_p1"],
    )  # has grad
    return entropy

def get_entropy(
    p0, 
    log_p0,
    p1, 
    log_p1,
):  # has grad
    # Entropy for binary actions: H(p) = -p*log(p) - (1-p)*log(1-p)
    entropy = -(p1 * log_p1 + p0 * log_p0).mean()  # Shape [b, N]
    return entropy
