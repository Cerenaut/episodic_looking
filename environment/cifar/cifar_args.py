import argparse
import json
import logging

logger = logging.getLogger(__name__)

class CifarArgs:

    EXPERIMENT_TYPE_PRETRAIN = "pretrain"
    EXPERIMENT_TYPE_FEW_SHOT = "few-shot"
    EXPERIMENT_TYPE_CONTINUAL = "continual"
    EXPERIMENT_TYPE_STREAMING = "streaming"
    EXPERIMENT_TYPE_EVALUATE = "evaluate"  # load the STM checkpoint and evaluate the four test sets once (no training)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=12,
        )

        parser.add_argument(
            "--evaluate-epochs",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.1,
        )

        parser.add_argument(
            "--sparsity",
            type=int,
            default=32,
        )

        parser.add_argument(
            "--experiment-type",
            choices=[
                CifarArgs.EXPERIMENT_TYPE_PRETRAIN, 
                CifarArgs.EXPERIMENT_TYPE_FEW_SHOT, 
                CifarArgs.EXPERIMENT_TYPE_CONTINUAL, 
                CifarArgs.EXPERIMENT_TYPE_STREAMING,
                CifarArgs.EXPERIMENT_TYPE_EVALUATE,
            ],
            default="few-shot",
        )

        parser.add_argument(
            "--max-instances",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--fine-classes",
            type=int,
            nargs="+",
            default=[3],
        )

        parser.add_argument(
            "--coarse-classes",
            type=int,
            nargs="+",
            default=[0,1],
        )

        # Variant / experiment-management options (added for the baselines comparison).
        parser.add_argument(
            "--actor-training",
            choices=["rl", "differentiable"],
            default="rl",
            help="rl: the paper's actor-critic (PPO-style) training of the bias policy. "
                 "differentiable: the actor's mean output is the bias, trained by back-propagating the "
                 "classifier's cross-entropy through the frozen LTM into the actor (no sampling, no critic).",
        )
        parser.add_argument(
            "--ltm-checkpoint",
            type=str,
            default=None,
            help="Frozen LTM checkpoint (default: the script's built-in path).",
        )
        parser.add_argument(
            "--stm-checkpoint",
            type=str,
            default=None,
            help="STM checkpoint written by --experiment-type pretrain and read by the other experiment types "
                 "(default: the script's built-in path).",
        )
        parser.add_argument(
            "--stm-checkpoint-out",
            type=str,
            default=None,
            help="continual only: prefix for STM checkpoints saved after each phase "
                 "(<prefix>_phase<fine class>.pth), for post-hoc analysis (analyze_stm_inspectability.py).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed torch (CPU and MPS/CUDA), numpy (env image sampling, dataset subsets) and the gym action spaces. "
                 "Default None = unseeded, the previous behaviour. Runs are then repeatable up to backend nondeterminism.",
        )
        parser.add_argument(
            "--run-root",
            type=str,
            default="./runs",
            help="Directory under which results are written (runs/cifar_100/<experiment>/<timestamp>/).",
        )
        parser.add_argument(
            "--eval-bias",
            choices=["sample", "mean"],
            default="sample",
            help="RL actor only: bias used during evaluation. sample = draw from the policy (paper behaviour); "
                 "mean = deterministic policy mean, as the differentiable actor always uses.",
        )
        parser.add_argument(
            "--evaluate-steps",
            type=int,
            default=1600,
            help="agent.step() calls per evaluation of one test set (1600 = 16 passes over its 200 images with the "
                 "sampled bias; 800 is enough for the deterministic mean bias: std about 0.012 per point).",
        )
        parser.add_argument(
            "--training-steps",
            type=int,
            default=4000,
            help="agent.step() calls per reported epoch (4000 = 8,000 image exposures at batch 16, 8-step episodes).",
        )

        args = parser.parse_args()
        logger.info("Parse args:")
        logger.info(json.dumps(vars(args), indent=4))
        return args

