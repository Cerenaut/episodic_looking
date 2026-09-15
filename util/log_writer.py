from torch.utils.tensorboard import SummaryWriter


class ScalarLogWriter:
    """
    Accumulates scalar quantities and periodically writes the mean value.
    """
    
    def __init__(
        self, 
        epoch_type:str,
    ):
        self.epoch_type = epoch_type
        self.reset()

    def reset(self):
        self.scalars = {}
        self.counts = {}

    def add_scalar(self, name:str, value:float):
        if name not in self.scalars.keys():
            self.scalars[name] = 0.0
            self.counts[name] = 0
        self.scalars[name] += value
        self.counts[name] += 1

    def write_scalars(self, time_index:int):
        for k, v in self.scalars.items():
            #print(f"Log TB K: {k} v: {v}")
            n = float(self.counts[k])
            mean = v / n

            key = self.get_series_name(
                self.epoch_type,
                k,
            )
            self._write_scalar(key, mean, time_index)

    def _write_scalar(
        self,
        name:str,
        value:float,
        time_index:int,
    ):
        raise NotImplementedError

    def get_series_name(self, epoch_type:str, key:str):
        series_name = f"{epoch_type} {key}"
        return series_name


class TensorboardLogWriter(ScalarLogWriter):
    """
    Writes logged values to Tensorboard
    """
    
    WRITER_TYPE = "tensorboard"
    
    def __init__(
        self, 
        epoch_type:str,
        log_path:str,
    ):
        super().__init__(epoch_type)
        self.writer = SummaryWriter(log_path)

    def _write_scalar(
        self,
        name:str,
        value:float,
        time_index:int,
    ):
        self.writer.add_scalar(
            name,
            value,
            time_index,
        )

class WandbLogWriter(ScalarLogWriter):
    """
    Writes logged values to Wandb.
    """
    
    WRITER_TYPE = "wandb"
    
    def __init__(
        self, 
        epoch_type:str,
        log_path:str,
    ):
        super().__init__(epoch_type)

    def _write_scalar(
        self,
        name:str,
        value:float,
        time_index:int,
    ):
        raise NotImplementedError
