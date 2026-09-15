import logging
import pickle
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

@dataclass
class CifarSharedMemoryNames:
    """
    Shared memory is used to avoid having a copy of each dataset in memory per 
    async environment. If we have a minibatch size > 1, each batch sample has its
    own copy of the environment. Async means separate processes for each env instance.
    """
    images: str | None = None
    labels_coarse: str | None = None
    labels_fine: str | None = None


class Cifar100Dataset(Dataset):
    """
    Dataset for processing the Cifar-100 dataset, which is described here:
    https://cave.cs.toronto.edu/kriz/cifar.html
    """

    LABEL_TYPE_FINE = "fine"
    LABEL_TYPE_COARSE = "coarse"

    def __init__(
        self,
        file_path,
        label_type,
        training:bool=True,
        exclude_classes_coarse:set[int]|None = None,
        exclude_classes_fine:set[int]|None = None,
        max_instances:int|None = None,
        shared_memory_names:CifarSharedMemoryNames|None = None,
        as_tensor:bool = True,
    ):
        self.max_instances = max_instances
        self.as_tensor = as_tensor
        self.shared_memory_names = shared_memory_names
        self.shared_memory_images = None
        self.shared_memory_labels_coarse = None
        self.shared_memory_labels_fine = None

        images, labels_coarse, labels_fine = self.get_data(file_path, training)

        # Optionally filter samples to only some classes
        self.exclude_classes_coarse = exclude_classes_coarse
        self.exclude_classes_fine = exclude_classes_fine        

        self.images, self.labels_coarse, self.labels_fine = self.filter_data(
            images, 
            labels_coarse, 
            labels_fine,
            self.exclude_classes_coarse,
            self.exclude_classes_fine,
        )

        # Optionally sample a subset of the data
        num_instances = len(self.images)
        if self.max_instances is None:
            pass
        else:  # Reduce to a finite number of instances
            self.images, self.labels_coarse, self.labels_fine = self.sample_data(
                self.images,
                self.labels_coarse,
                self.labels_fine,
                self.exclude_classes_coarse,
                self.max_instances,
            )

        # Select which labels to emit as target classes
        if label_type == Cifar100Dataset.LABEL_TYPE_FINE:
            self.labels = self.labels_fine
            self.num_classes = 100

        elif label_type == Cifar100Dataset.LABEL_TYPE_COARSE:
            self.labels = self.labels_coarse
            self.num_classes = 20

        else:
            raise ValueError(
                f"Unknown label_type: {label_type}"
            )

        # Preprocess images to make them ready to consume
        self.images = self.images.reshape(
            -1, 3, 32, 32
        ).astype(np.float32) / 255.0

        # Reduce memory footprint during use
        self.images, self.labels_coarse, self.labels_fine = self.create_shared_memory(
            self.images,
            self.labels_coarse,
            self.labels_fine,
        )
        
        # Optionally skip the from_numpy - RL env requires numpy, torch prefers tensor.
        if self.as_tensor:
            logger.info("Converting dataset to tensor...")
            self.images = torch.from_numpy(
                self.images
            ).float()

            self.labels = torch.tensor(
                self.labels,
                dtype=torch.long,
            )

    @staticmethod 
    def get_image_shape() -> list[int]:
        return [3, 32, 32]  # C, H, W
    
    def get_shared_memory_names(self) -> CifarSharedMemoryNames:
        return self.shared_memory_names
    
    def create_shared_memory(
        self, 
        images:np.ndarray,
        labels_coarse:list[int],
        labels_fine:list[int],

    ) -> tuple[np.ndarray, list[int], list[int]]:

        if self.shared_memory_names is None:
            self.shared_memory_names = CifarSharedMemoryNames(
                images = None,
                labels_coarse = None,
                labels_fine = None,
            )

        self.shared_memory_images, shared_images = Cifar100Dataset._create_shared_memory(
            shared_memory_name = self.shared_memory_names.images, 
            array = images,
        )
        self.shared_memory_names.images = self.shared_memory_images.name

        labels_coarse_array = np.asarray(labels_coarse, dtype=np.int64)
        self.shared_memory_labels_coarse, shared_labels_coarse_array = Cifar100Dataset._create_shared_memory(
            shared_memory_name = self.shared_memory_names.labels_coarse, 
            array = labels_coarse_array,
        )
        self.shared_memory_names.labels_coarse = self.shared_memory_labels_coarse.name
        shared_labels_coarse = shared_labels_coarse_array.tolist()

        labels_fine_array = np.asarray(labels_fine, dtype=np.int64)
        self.shared_memory_labels_fine, shared_labels_fine_array = Cifar100Dataset._create_shared_memory(
            shared_memory_name = self.shared_memory_names.labels_fine, 
            array = labels_fine_array,
        )
        self.shared_memory_names.labels_fine = self.shared_memory_labels_fine.name
        shared_labels_fine = shared_labels_fine_array.tolist()

        # Discard original copy of buffer in favour of shared
        return shared_images, shared_labels_coarse, shared_labels_fine

    @staticmethod
    def _create_shared_memory(shared_memory_name:str|None, array:np.ndarray) -> tuple[shared_memory.SharedMemory, np.ndarray]:
        if shared_memory_name is None:
            shared_object = shared_memory.SharedMemory(
                create=True,
                size=array.nbytes,
            )
            shared_array = np.ndarray(
                array.shape,
                dtype=array.dtype,
                buffer=shared_object.buf,
            )  # forget original images, keep this (single, shared) copy.

            shared_array[:] = array  # Copy contents to shared buffer
            #logger.info(f"Created shared array named: {shared_object.name}")

        else:
            shared_object = shared_memory.SharedMemory(
                name=shared_memory_name,
            )
            shared_array = np.ndarray(
                array.shape,
                dtype=array.dtype,
                buffer=shared_object.buf,
            )  # forget original images, keep this (single, shared) copy.
        
            # No copy reqd
            #logger.info(f"Using shared array named: {shared_memory_name}")
        return shared_object, shared_array

    def get_data(self, file_path:str, training:bool):
        #logger.info(f"Loading data file: '{file_path}' training?:{training}...")
        root = Path(file_path)
        filename = "train" if training else "test"
        path = root / filename

        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")

        # Raw CIFAR data is:
        # [N, 3072] = 3 x 32 x 32
        images = data[b"data"]  # a bit slow but I have to read the file anyways
        labels_coarse = data[b"coarse_labels"]
        labels_fine = data[b"fine_labels"]
        return images, labels_coarse, labels_fine

    def filter_data(
        self,
        images:np.ndarray,
        labels_coarse:list[int],
        labels_fine:list[int],
        exclude_classes_coarse: set[int] | None = None,
        exclude_classes_fine: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        labels_coarse = np.asarray(labels_coarse)
        labels_fine = np.asarray(labels_fine)

        include = np.ones(len(images), dtype=bool)

        if exclude_classes_coarse is not None:
            include &= ~np.isin(
                labels_coarse,
                list(exclude_classes_coarse),
            )

        if exclude_classes_fine is not None:
            include &= ~np.isin(
                labels_fine,
                list(exclude_classes_fine),
            )

        # Copy containing only the retained samples.
        included_images = images[include].copy()
        included_labels_coarse = [
            int(x) for x in labels_coarse[include]
        ]

        included_labels_fine = [
            int(x) for x in labels_fine[include]
        ]

        return included_images, included_labels_coarse, included_labels_fine

    def sample_data(
        self,
        images:np.ndarray,
        labels_coarse:list[int],
        labels_fine:list[int],
        exclude_classes_coarse: set[int] | None,
        max_instances:int,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        
        # Given the excluded coarse labels, figure out the included coarse labels.
        # Since there are known to be 20 of them
        exclude_classes_coarse = exclude_classes_coarse or set()  # or empty set
        num_coarse_classes = 20
        coarse_classes = set(range(num_coarse_classes)) - exclude_classes_coarse

        labels_coarse_array = np.asarray(labels_coarse)
        selected_instance_indices = []

        # Select self.max_instances from each coarse class.
        for coarse_class in coarse_classes:
            coarse_class_indices = np.flatnonzero(
                labels_coarse_array == coarse_class
            )  # which indices are the desired coarse class

            num_instances = len(coarse_class_indices)
            if num_instances < max_instances:
                raise ValueError(
                    f"Coarse class {coarse_class} only has "
                    f"{len(coarse_class_indices)} instances, but "
                    f"max_instances={max_instances} requested."
                )

            selected_indices = np.random.choice(
                coarse_class_indices,
                size=self.max_instances,
                replace=False,
            )

            selected_instance_indices.extend(selected_indices)

        self.selected_instance_indices = np.asarray(
            selected_instance_indices,
            dtype=np.int64,
        )

        # Extract selected images and labels.
        selected_images = images[self.selected_instance_indices].copy()

        selected_labels_coarse = [
            labels_coarse[i]
            for i in self.selected_instance_indices
        ]

        selected_labels_fine = [
            labels_fine[i]
            for i in self.selected_instance_indices
        ]

        return (
            selected_images,
            selected_labels_coarse,
            selected_labels_fine,
        )

    def get_classes(self) -> dict:
        counts = {}
        for index, _ in enumerate(self.images):
            label_coarse = self.labels_coarse[index]
            label_fine = self.labels_fine[index]
            if label_coarse not in counts:
                counts[label_coarse] = {}
            
            counts_coarse = counts[label_coarse]
            if label_fine not in counts_coarse.keys():
                counts_coarse[label_fine] = 0
            counts_coarse[label_fine] += 1
        return counts

    def get_num_classes(self) -> int:
        return self.num_classes
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.images[index],
            self.labels[index],
        )

    @staticmethod
    def get_coarse_classes_excluded(coarse_classes):
        num_classes = 20
        exclude_classes_coarse = set()
        for i in range(num_classes):
            if i not in coarse_classes:
                exclude_classes_coarse.add(i)
        return exclude_classes_coarse
    
    @staticmethod
    def get_fine_class_set(fine_class:int):
        if fine_class == 1:
            return {
                4, # 0
                1, # 1
                54, # 2
                9, # 3
                0, # 4
                22, # 5
                5, # 6
                6, # 7
                3, # 8
                12, # 9
                23, # 10
                15, # 11
                34, # 12
                26, # 13
                2, #14
                27, #15
                36, # 16
                47, # 17
                8, # 18
                41, # 19
            }
        elif fine_class == 2:
            return {
                30, # 0
                32, # 1
                62, # 2
                10, # 3
                51, # 4
                39, # 5
                20, # 6
                7, # 7
                42, # 8
                17, # 9
                33, # 10
                19, # 11
                63, # 12
                45, # 13
                11, #14
                29, #15
                50, # 16
                52, # 17
                13, # 18
                69, # 19
            }
        elif fine_class == 3:
            return {
                55,  # 0
                67,  # 1
                70,  # 2
                16,  # 3
                53,  # 4
                40, # 5
                25, # 6
                14, # 7
                43, # 8
                37, # 9
                49, # 10
                21, # 11
                64, # 12
                77, # 13
                35, # 14
                44, # 15
                65, # 16
                56, # 17
                48, # 18
                81, # 19
            }    
        elif fine_class == 4:
            return {
                72,  # 0
                73,  # 1
                82,  # 2
                28,  # 3
                57,  # 4
                86, # 5
                84, # 6
                18, # 7
                88, # 8
                68, # 9
                60, # 10
                31, # 11
                66, # 12
                79, # 13
                46, # 14
                78, # 15
                74, # 16
                59, # 17
                58, # 18
                85, # 19
            }    
        elif fine_class == 5:
            return {
                95,  # 0
                91,  # 1
                92,  # 2
                61,  # 3
                83,  # 4
                87, # 5
                94, # 6
                24, # 7
                97, # 8
                76, # 9
                71, # 10
                38, # 11
                75, # 12
                99, # 13
                98, # 14
                93, # 15
                80, # 16
                96, # 17
                90, # 18
                89, # 19
            }    
        raise ValueError("Fine class set not recognized.")

    @staticmethod
    def get_fine_classes(fine_classes:list[int]):
        all_fine_classes = None
        for fine_class in fine_classes:
            fine_class_set = Cifar100Dataset.get_fine_class_set(fine_class)
            if all_fine_classes is None:
                all_fine_classes = fine_class_set
            else:
                all_fine_classes = all_fine_classes.union(fine_class_set)
        return all_fine_classes
