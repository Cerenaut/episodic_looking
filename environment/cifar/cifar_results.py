import os
import re

import pandas as pd


class CifarResults:

    """
    Standardized file format for Cifar evaluate results.
    """

    def __init__(self, run_path:str, suffix:str):
        self.run_path = run_path
        self.suffix = suffix

    def get_file_name(self) -> str:
        return os.path.join(self.run_path, f"results_{self.suffix}.txt")

    def clear_file(self):
        file_name = self.get_file_name()
        with open(file_name, "w"):
            pass

    def append_file(self, text):
        file_name = self.get_file_name()
        with open(file_name, "a") as f:
            f.write(text)

    def get_line(
        coarse_classes:str,
        fine_classes:str,
        mode:str,
        epoch:int,
        accuracy:float,            
    ) -> str:
        results_line = f"{coarse_classes}, {fine_classes}, {mode}, {epoch!s}, {accuracy}\n"
        return results_line

    def append_line(
        self,
        coarse_classes:str,
        fine_classes:str,
        mode:str,
        epoch:int,
        accuracy:float,            
    ):
        results_line = CifarResults.get_line(coarse_classes, fine_classes, mode, epoch, accuracy)
        self.append_file(results_line)

    @staticmethod
    def read_results_file(file_name:str) -> pd.DataFrame:
        rows = []

        max_epoch_relative = 0
        max_epoch = 0
        epoch_offset = 0

        with open(file_name) as f:
            for line in f:
                parts = re.split(r'\s*,\s*(?=(?:[^\[\]]*\[[^\[\]]*\])*[^\[\]]*$)', line.strip())

                mode = parts[2]
                if mode == "training":
                    continue
                
                epoch_relative = int(parts[3])
                max_epoch_relative = max(max_epoch_relative, epoch_relative)  # largest value ever seen
                epoch = epoch_relative + epoch_offset
                if epoch < max_epoch:  # e.g. 12 -> 0
                    epoch_offset += (max_epoch_relative +1)
                    epoch = epoch_relative + epoch_offset

                #print(f"Rel:{epoch_relative} Off:{epoch_offset} Epoch:{epoch} max Rel:{max_epoch_relative} max Epoch:{max_epoch}")

                rows.append([
                    re.sub(r"[\[\]']", "", parts[0]).replace(", ", ","),
                    re.sub(r"[\[\]']", "", parts[1]).replace(", ", ","),
                    mode,
                    epoch +1,  # +1 because evaluated after epoch
                    float(parts[4]),
                ])

                max_epoch = max(epoch, max_epoch)

        fine_class = "Fine class"
        df = pd.DataFrame(
            rows,
            columns=[
                "Coarse classes",
                fine_class,
                "Mode",
                "Epoch",
                "Accuracy",
            ],
        )
        return df

    @staticmethod
    def read_results_files(file_names:list[str]) -> list[pd.DataFrame]:
        results = []
        for file_name in file_names:
            df = CifarResults.read_results_file(file_name)
            df_evaluate = df[df["Mode"] == "evaluate"]
            results.append(df_evaluate)
        return results
