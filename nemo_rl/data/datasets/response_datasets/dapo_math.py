# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class DAPOMath17KDataset(RawDataset):
    """Simple wrapper around the DAPO Math 17K dataset with train split.

    Args:
        deduplicate: Keep only the first row of each distinct prompt. The upload
            repeats each of its ~17.4k problems about 100 times (1.79M rows).
    """

    def __init__(self, deduplicate: bool = False, **kwargs) -> None:
        self.task_name = "DAPOMath17K"

        # load from huggingface
        self.dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        if deduplicate:
            self.dataset = self._deduplicate(self.dataset)

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    @staticmethod
    def _deduplicate(dataset: Dataset) -> Dataset:
        prompts = pc.struct_field(
            pc.list_element(dataset.data.column("prompt"), 0), "content"
        )
        codes = pc.index_in(prompts, value_set=pc.unique(prompts)).to_numpy()
        first_rows = np.sort(np.unique(codes, return_index=True)[1])
        print(
            f"DAPOMath17K: kept {len(first_rows)} distinct prompts of {len(dataset)} rows"
        )
        return dataset.select(first_rows)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": data["prompt"][0]["content"],
                },
                {
                    "role": "assistant",
                    "content": data["reward_model"]["ground_truth"],
                },
            ],
            "task_name": self.task_name,
        }


class DAPOMathAIME2024Dataset(DAPOMath17KDataset):
    def __init__(self, **kwargs) -> None:
        """Initialize the DAPO Math AIME 2024 dataset with train split."""
        self.task_name = "DAPOMathAIME2024"

        # load from huggingface
        self.dataset = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
