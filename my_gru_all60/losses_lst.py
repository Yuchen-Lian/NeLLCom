# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

class MyLoss_lst:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_input: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        eq = torch.eq(message, labels).sum(dim=1)
        total_steps = labels.shape[1]
        acc = torch.eq(eq, total_steps) * 1.0

        for i in range(receiver_output.shape[1]):
            step_output = receiver_output[:, i, :].contiguous().view(batch_size, -1)
            l = self.criterion(step_output, labels[:, i])
            accumulate_loss += l
        return accumulate_loss, {"acc": acc}
