# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

# list all possible utterances
language_dict = {'fix': 0, 'fix_mk': 1, 'free_mk': 2, 'fix_op': 3, 'free_op': 4}
inv_language_dict = {v: k for k, v in language_dict.items()}


def acc_eval_compute(labels, aux_inpute, message):

    language = inv_language_dict[aux_inpute['language'][0].item()]
    total_steps = labels.shape[1]-1
    eq1 = torch.eq(message, labels[:, 1:]).sum(dim=1)
    corr1 = torch.eq(eq1, total_steps)

    if language == 'fix_mk' or language == 'fix':
        acc = corr1 * 1.0
    elif language == 'free_mk':

        f2r = [0, 2, 3, 1, 4, 5]
        r2f = [0, 3, 1, 2, 4, 5]

        mk_expand = aux_inpute['mk_idx'].unsqueeze(1).expand(-1, labels.size(1))
        mk_locate = torch.nonzero(torch.eq(labels, mk_expand))
        new_labels = []

        if mk_locate.size(0) != labels.size(0):
            acc = corr1 * 1.0
            return acc

        for i, j in mk_locate:
            if j == 2:
                new_l = labels[i][r2f]
            elif j == 3:
                new_l = labels[i][f2r]
            else:
                new_l = None
            new_labels.append(new_l)
        new_labels = torch.stack(new_labels)

        eq2 = torch.eq(message, new_labels[:, 1:]).sum(dim=1)
        corr2 = torch.eq(eq2, total_steps)
        print(corr2.sum())
        acc = torch.logical_or(corr1, corr2) * 1.0

    elif language == 'fix_op':
        acc = corr1*1.0

    elif language == 'free_op':
        acc = corr1*1.0

    return acc


class MyLoss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        speaker_output: torch.Tensor,
        speaker_message: torch.Tensor,
        listener_output: torch.Tensor,
        listener_prediction: torch.Tensor,
        aux_input: Dict[str, torch.Tensor],
        is_training: True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        batch_size = sender_input.size(0)
        total_steps = labels.shape[1]

        n_attributes = labels.size(1)
        n_values = listener_output.size(-1)
        listener_output = listener_output.view(batch_size * total_steps, n_values)
        correct_samples = (
            (listener_prediction == labels).view(batch_size, -1).detach()
        )
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        labels = labels.view(batch_size * n_attributes)
        loss = F.cross_entropy(listener_output, labels, reduction="none")
        loss = loss.view(batch_size, -1).mean(dim=1)

        return loss, {"acc": acc}

