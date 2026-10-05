import argparse
import json
import logging

logger = logging.getLogger(__name__)

class CifarArgs:
    """
    Command-line arguments of the STM (cifar_main_stm_training.py) and LTM-only (cifar_main_ltm_fine_tuning.py)
    scripts. Each script declares only the flags it honours: the flags both use (add_common_args) plus its own
    (add_stm_args, add_ltm_args), with the same names and defaults in both.
    """

    EXPERIMENT_TYPE_PRETRAIN = "pretrain"
    EXPERIMENT_TYPE_FEW_SHOT = "few-shot"
    EXPERIMENT_TYPE_CONTINUAL = "continual"
    EXPERIMENT_TYPE_STREAMING = "streaming"
    EXPERIMENT_TYPE_EVALUATE = "evaluate"  # load the STM checkpoint and evaluate the four test sets once (no training)

    @staticmethod
    def parse_stm_args(argv:list[str]|None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        CifarArgs.add_common_args(parser)
        CifarArgs.add_stm_args(parser)
        return CifarArgs._parse(parser, argv)

    @staticmethod
    def parse_ltm_args(argv:list[str]|None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        CifarArgs.add_common_args(parser)
        CifarArgs.add_ltm_args(parser)
        return CifarArgs._parse(parser, argv)

    @staticmethod
    def _parse(parser:argparse.ArgumentParser, argv:list[str]|None) -> argparse.Namespace:
        args = parser.parse_args(argv)
        logger.info("Parse args:")
        logger.info(json.dumps(vars(args), indent=4))
        return args

    @staticmethod
    def add_common_args(parser:argparse.ArgumentParser):
        """Flags both scripts honour."""
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
            help="STM: what to run. LTM-only: names the run only (it always trains on the fine classes in order).",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
        )

        parser.add_argument(
            "--evaluate-epochs",
            type=int,
            default=1,
            help="Evaluate at every epoch whose index is a multiple of this, and at the last epoch (few-shot STM; every "
                 "phase for LTM-only). 1 = every epoch. STM pre-training skips the last-epoch rule, as before.",
        )

        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.1,
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

        parser.add_argument(
            "--ltm-checkpoint",
            type=str,
            default=None,
            help="Frozen LTM checkpoint (STM), or the LTM to fine-tune (LTM-only) (default: the script's built-in path).",
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

    @staticmethod
    def add_stm_args(parser:argparse.ArgumentParser):
        """Flags of the STM script only."""
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Epochs (per phase in continual) of --training-steps steps each. Default None = 12.",
        )

        parser.add_argument(
            "--sparsity",
            type=int,
            default=32,
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
            help="STM checkpoints after training, for post-hoc analysis (analyze_stm_inspectability.py). continual: a "
                 "prefix, one checkpoint per phase (<prefix>_phase<fine class>.pth); few-shot: the checkpoint's path.",
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
            "--eval-batch-size",
            type=int,
            default=None,
            help="Parallel environments in evaluation, independent of --batch-size (separate environments). "
                 "Default None = --batch-size, sharing the training environments as before.",
        )
        parser.add_argument(
            "--eval-sweep",
            action="store_true",
            help="Evaluate with one deterministic pass over every image of each test and validation set (one "
                 "8-step episode per image), instead of --evaluate-steps steps on images drawn with replacement.",
        )
        parser.add_argument(
            "--eval-record-images",
            action="store_true",
            help="With --eval-sweep: also write each image's result (1/0, in dataset order) per evaluation to "
                 "results_<type>_images.txt.",
        )

        parser.add_argument(
            "--ltm-obs-cache",
            choices=["on", "off", "check"],
            default="on",
            help="Reuse each step's second LTM pass as the next step's first while no episode ended (same image "
                 "and bias, frozen LTM), saving 7 of 16 LTM passes per episode; results are bitwise identical. "
                 "off = recompute every pass (the original code path); check = recompute and fail unless identical.",
        )

    @staticmethod
    def add_ltm_args(parser:argparse.ArgumentParser):
        """Flags of the LTM-only script only."""
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Epochs per phase. Default None = int(6250 / instances per epoch) (12 at 500 instances), which it "
                 "previously always used.",
        )
        parser.add_argument(
            "--evaluate-points",
            type=int,
            default=None,
            help="Evaluate at about this many log-spaced epochs of each phase, first and last included "
                 "(CifarResults.evaluation_epochs), instead of --evaluate-epochs. Default None.",
        )
        parser.add_argument(
            "--loader-workers",
            type=int,
            default=2,
            help="DataLoader worker processes. Workers are re-spawned every epoch (macOS), which dominates short "
                 "epochs; 0 loads in the main process. The data order comes from the samplers, not the workers.",
        )
        parser.add_argument(
            "--bn-mode",
            choices=["train", "frozen"],
            default="train",
            help="LTM-only fine-tuning: batch normalization during training. train = training mode (batch statistics, "
                 "running statistics updated, affine parameters trained; the draft's behaviour and the default). "
                 "frozen = eval mode throughout (running statistics fixed) and the BN affine parameters not trained; "
                 "all other weights train as before. Used for single-stream (minibatch 1), decided 6 Oct 2026.",
        )
