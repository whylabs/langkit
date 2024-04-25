from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class AdditionalData:
    additional_data_path: str

    @lru_cache
    def encode_additional_data(self) -> torch.Tensor:
        if ".npy" in self.additional_data_path:
            return torch.tensor(np.load(self.additional_data_path))
        else:
            raise ValueError("Unsupported additional data format. Use a .npy file from numpy.")
