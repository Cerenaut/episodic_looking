import logging
from dataclasses import dataclass

from util.config import dataclass_write_json_file
from util.log import get_run_path, periodic
from util.log_writer import ScalarLogWriter, TensorboardLogWriter, WandbLogWriter

logger = logging.getLogger(__name__)

@dataclass
class InstrumentationConfig:

    # Console log
    steps_console_update: int = 100  # How often to update console with progress

    # Logging
    log_type: str = "tensorboard"
    log_path: str = "./runs"
    log_prefix: str = ""
    log_period: int = 100

class Instrumentation:

    """
    A class which manages modes (e.g. training, evaluate), global steps in each mode, 
    and periodic log-writing for each mode so that we can log running averages of highly
    volatite quantities.
    """

    MODE_TRAINING = "training"
    MODE_EVALUATE = "evaluate"

    def __init__(self, config:InstrumentationConfig, modes:list[str] = None):
        self.config = config
        self.run_path = get_run_path(
            prefix = self.config.log_prefix, 
            path = self.config.log_path,
        )

        if modes is None:
            modes = Instrumentation.get_default_modes()
        self.modes = modes
        self.mode = None

        self.reset_steps()
        self._create_log_writers()

    @staticmethod
    def get_default_modes():
        return [
            Instrumentation.MODE_TRAINING,
            Instrumentation.MODE_EVALUATE,
        ]

    def write_config(self, config, file_name:str = "config.json"):
        """
        Utility function to persist a config object to a file in the same directory as the logs.
        """
        dataclass_write_json_file(
            obj = config,
            file_path = self.run_path, 
            file_name = file_name,
        )

    # MODES    
    def get_mode(self) -> str|None:
        return self.mode
    
    def is_mode_training(self):
        return self.mode == Instrumentation.MODE_TRAINING

    def is_mode_evaluate(self):
        return self.mode == Instrumentation.MODE_EVALUATE

    def set_mode(self, mode:str):
        self.mode = mode
                  
    # STEPS
    def reset_steps(self):
        self.global_steps = {}
        for mode in self.modes:
            self.global_steps[mode] = 0

    def get_steps(self, mode:str = None) -> int|None:
        if mode is None:
            mode = self.mode
        return self.global_steps.get(mode)
    
    def increment_steps(self, mode:str = None) -> int:
        if mode is None:
            mode = self.mode

        step = self.global_steps[mode]
        step += 1
        self.global_steps[mode] = step
        return step

    def print_steps(self, num_steps:int = 0):
        step = self.get_steps()
        if (step < 100) or (step % self.config.steps_console_update == 0):
            if num_steps == 0:
                num_steps = "n/a"
            logging.info(f"Step: {step} of {num_steps}")

    # LOG WRITERS
    def get_log_writer(self):
        return self._get_log_writer(self.mode)
    
    def flush_log_writers(self, force:bool=False):
        # Accumulate, average (for smoothing) and periodically write log values:
        global_step = self.get_steps(self.mode)
        if force or periodic(t=global_step, period=self.config.log_period):
            self._flush_log_writer(self.mode, global_step)

    def _create_log_writers(self):
        self.log_writers = {}
        for mode in self.modes:
            self.log_writers[mode] = self._create_log_writer(mode)

    def _create_log_writer(self, epoch_type:str) -> ScalarLogWriter:
        if self.config.log_type == TensorboardLogWriter.WRITER_TYPE:
            return TensorboardLogWriter(
                epoch_type = epoch_type,
                log_path = self.run_path,
            )
        elif self.config.log_type == WandbLogWriter.WRITER_TYPE:
            return WandbLogWriter(
                epoch_type = epoch_type,
                log_path = self.run_path,
            )
        raise NotImplementedError 

    def _get_log_writer(self, epoch_type:str) -> ScalarLogWriter:
        return self.log_writers.get(epoch_type)

    def _flush_log_writer(self, mode:str, global_step:int):
        writer = self._get_log_writer(mode)
        writer.write_scalars(time_index = global_step)
        writer.reset()  # clear logged values to allow another block to be accumulated


