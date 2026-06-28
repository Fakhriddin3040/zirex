from typing import TypedDict

import numpy as np
import numpy.typing as npt
from torch import Tensor

# Audio is carried as a 1-D array of float samples (engine output is float32).
FloatArray = npt.NDArray[np.float32]


class TokenizerOutput(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
