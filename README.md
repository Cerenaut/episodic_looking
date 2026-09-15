# Episodic looking - using reinforcement learning to control perception enables continual, streaming, online learning without forgetting

## Abstract
There is a fundamental contradiction between the desire for models that can generalize and models that can learn rapidly and effectively from a single thread of individual, real-world experience. Capture of statistical regularities creates entanglement between latent variables, causing destructive interference in continual or online, streaming learning settings. But without statistical regularities, generalization cannot occur, leaving the model unable to overcome input variance. This paper proposes a solution based on the Complementary Learning System architecture (CLS), in which a Short-Term Memory (STM) gives up generalization in exchange for continual learning without interference, working in a space of robust, generalized perceptions produced by a slow, statistical Long-Term Memory (LTM).

We show that on an image classification task, the resulting CLS model is resistant to interference in a continual learning setting while achieving comparable or better generalization and transfer performance than a baseline ResNet LTM alone. The CLS model is also successful in single-stream, online and few-shot learning scenarios.

![alt text](/readme.png?raw=true)

**Above:** Adding an episodic short-term memory (STM) to a baseline ResNet long-term memory (LTM) dramatically improves generalization and transfer performance in a continual learning task on the Cifar-100 dataset, while also gaining resistance to interference between earlier and later memories. In the image above, both models are sequentially exposed to fine-classes 3, 4, and 5 of a pair of Cifar-100 coarse classes. The grey vertical bars show when training shifts from fine-class 3 to fine-class 4 and then again when training shifts to fine-class 5. The proposed CLS/STM architecture (left) rapidly learns a generalized solution; it is unaffected by the transition from classes 3 to 4 or 4 to 5. The LTM-only model, without the STM, suffers significant interference when the training class changes, demonstrated by drops in evaluation accuracy on the classes not currently being trained.

All experiments are online, with the model updated each step of each episode. Single-stream experiments also have minibatch size = 1, i.e. replicating the learning conditions of a single mobile robot who must continually learn about its changing world.

Arxiv preprint: [https://TODO](https://TODO)

## Getting started
Download the Cifar-100 dataset from [https://cave.cs.toronto.edu/kriz/cifar.html](https://cave.cs.toronto.edu/kriz/cifar.html); it is required by the code.

### Entry points
* [test_sparse_distributed_model.ipynb](test_sparse_distributed_model.ipynb) - Minimal unit test of the sparse distributed memory module (without RL bits) in case you want to use it elsewhere
* [cifar_ltm_pretraining.ipynb](cifar_ltm_pretraining.ipynb) - Pretrain the LTM as in the paper
* [cifar_main_ltm_fine_tuning.py](cifar_main_ltm_fine_tuning.py) - Fine-tune the LTM; continual learning, few-shot and single-stream settings
* [cifar_main_stm_training.py](cifar_main_stm_training.py) - Pre-train and then fine-tune the STM (both are identical); continual learning, few-shot and single-stream settings

### Key bits of code
* [environment/cifar](environment/cifar) - Python code for the entire Cifar-100 dataset integration, RL Agent and RL Environment.
* [agent/episodic_agent.py](agent/episodic_agent.py) - Base class that sets up the RL loop for this and other environments.
* [model/sparse](model/sparse) - Sparse activation model being used in the Cifar-100 model as a STM.

There are also notebooks for reproducing the plots and other results shown in the paper. Some example bash scripts are provided to show how to configure the LTM and STM fine-tuning scripts. 
