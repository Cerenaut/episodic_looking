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
            default=None,
            help="STM: epochs (per phase in continual), default 12. LTM-only: epochs per phase, default "
                 "int(6250 / instances per epoch) (12 at 500 instances), which it previously always used.",
        )

        parser.add_argument(
            "--evaluate-epochs",
            type=int,
            default=1,
            help="Evaluate at every epoch whose index is a multiple of this, and at the last epoch (few-shot STM; every "
                 "phase for LTM-only). 1 = every epoch. STM pre-training skips the last-epoch rule, as before.",
        )
        parser.add_argument(
            "--loader-workers",
            type=int,
            default=2,
            help="LTM-only: DataLoader worker processes. Workers are re-spawned every epoch (macOS), which dominates "
                 "short epochs; 0 loads in the main process. The data order comes from the samplers, not the workers.",
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
            help="Frozen LTM checkpoint (STM), or the LTM to fine-tune (LTM-only) (default: the script's built-in path).",
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
            "--pretrain-coarse-classes",
            type=int,
            nargs="+",
            default=None,
            help="pretrain only: coarse classes whose fine classes 1,2 the STM is pre-trained on (e.g. 0..19 = all), "
                 "while evaluation stays on --coarse-classes. Default None = --coarse-classes.",
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

        parser.add_argument(
            "--val-holdout",
            type=int,
            default=0,
            help="Validation split: hold out this many training images per fine class (Cifar100Dataset."
                 "get_validation_mask, the same images as the head script's --val-holdout), train on the rest, and "
                 "evaluate the held-out images alongside the test sets, into results_<type>_val.txt. 0 = no split.",
        )
        parser.add_argument(
            "--split-seed",
            type=int,
            default=0,
            help="Seed of the validation split (not the run seed).",
        )
        parser.add_argument(
            "--eval-batch-size",
            type=int,
            default=None,
            help="STM: parallel environments in evaluation, independent of --batch-size (separate environments). "
                 "Default None = --batch-size, sharing the training environments as before.",
        )
        parser.add_argument(
            "--eval-sweep",
            action="store_true",
            help="STM: evaluate with one deterministic pass over every image of each test and validation set (one "
                 "8-step episode per image), instead of --evaluate-steps steps on images drawn with replacement.",
        )
        parser.add_argument(
            "--eval-record-images",
            action="store_true",
            help="STM, with --eval-sweep: also write each image's result (1/0, in dataset order) per evaluation to "
                 "results_<type>_images.txt.",
        )

        args = parser.parse_args()
        logger.info("Parse args:")
        logger.info(json.dumps(vars(args), indent=4))
        return args

