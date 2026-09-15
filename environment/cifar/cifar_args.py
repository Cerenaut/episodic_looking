import argparse
import json
import logging

logger = logging.getLogger(__name__)

class CifarArgs:

    EXPERIMENT_TYPE_PRETRAIN = "pretrain"
    EXPERIMENT_TYPE_FEW_SHOT = "few-shot"
    EXPERIMENT_TYPE_CONTINUAL = "continual"
    EXPERIMENT_TYPE_STREAMING = "streaming"

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

        args = parser.parse_args()
        logger.info("Parse args:")
        logger.info(json.dumps(vars(args), indent=4))
        return args

