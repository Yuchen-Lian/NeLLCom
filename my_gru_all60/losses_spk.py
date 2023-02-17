# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import numpy as np
import torch
from egg.zoo.my_gru_all60.utils import cutting_length

# list all possible utterances
language_dict = {'fix': 0, 'fix_mk': 1, 'free_mk': 2, 'fix_op': 3, 'free_op': 4}
inv_language_dict = {v: k for k, v in language_dict.items()}


def vari_len_acc_compute(labels, message, valid_len):

    batch_size = labels.size(0)

    corr_ = []
    for i in range(batch_size):
        valid_l = valid_len[i]
        eq = torch.eq(message[i, :valid_l], labels[i, 1:valid_l+1]).sum()
        corr_.append(torch.eq(eq, valid_l))
    corr = torch.stack(corr_)

    return corr


def acc_eval_compute(labels, aux_inpute, message, valid_len, pad_id):

    language = inv_language_dict[aux_inpute['language'][0].item()]

    corr1 = vari_len_acc_compute(labels, message, valid_len)
    # print(f'corr1:{(corr1*1.0).sum()}')

    mk_idx = aux_inpute['mk_idx'][0].item()

    if language == 'fix_mk' or language == 'fix':
        acc = corr1 * 1.0
    elif language == 'free_mk':

        f2r = [0, 2, 3, 1, 4, 5]
        r2f = [0, 3, 1, 2, 4, 5]

        mk_locate = torch.nonzero(torch.eq(labels, mk_idx))
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

        corr2 = vari_len_acc_compute(new_labels, message, valid_len)
        print(corr2.sum())
        acc = torch.logical_or(corr1, corr2) * 1.0

    elif language == 'fix_op':

        f2fm = [0, 1, 2, '<mk>', 3, 4]
        fm2f = [0, 1, 2, 4, 5, '<pad>']

        padded_len = labels.size(1)

        mk_locate = torch.nonzero(torch.eq(labels, mk_idx))
        has_mk = set(mk_locate[:, 0].tolist())
        not_has_mk = set(range(labels.size(0))) - has_mk

        new_labels = [None] * labels.size(0)
        new_lengths = [None] * labels.size(0)
        for i, j in mk_locate:
            # fix_mk to fix
            lab = labels[i].tolist()
            new_l = lab
            new_l.pop(j)
            new_l_len = len(new_l) - 1
            new_l.extend([pad_id]*(padded_len - (new_l_len + 1)))
            new_labels[i] = new_l
            new_lengths[i] = new_l_len

        for i in not_has_mk:
            # fix to fix_mk
            mk_location = 3
            lab = labels[i].tolist()
            new_l = lab
            new_l.insert(mk_location, mk_idx)
            new_l = new_l[:padded_len]
            new_l_len = len(new_l) - 1
            new_labels[i] = new_l
            new_lengths[i] = new_l_len

        # print(new_labels[0])
        tmp = np.array(new_labels)
        new_labels = torch.tensor(tmp)

        new_labels = torch.tensor(np.array(new_labels))
        new_lengths = torch.tensor(np.array(new_lengths))
        corr2 = vari_len_acc_compute(new_labels, message, new_lengths)
        # print(corr2.sum())
        acc = torch.logical_or(corr1, corr2) * 1.0

    elif language == 'free_op':

        m2fm = [0, 1, 2, '<mk>', 3, 4]
        m2fm = [0, 1, 2, '<mk>', 3, 4]
        m2fm = [0, 1, 2, '<mk>', 3, 4]
        fm2f = [0, 1, 2, 4, 5, '<pad>']

        f2r = [0, 2, 3, 1, 4, 5]
        r2f = [0, 3, 1, 2, 4, 5]

        fnm2rnm = [0, 2, 1, 3, 4, 5]

        padded_len = labels.size(1)

        mk_locate = torch.nonzero(torch.eq(labels, mk_idx))
        has_mk = set(mk_locate[:, 0].tolist())
        not_has_mk = set(range(labels.size(0))) - has_mk

        new_labels1, new_labels2, new_labels3 = [None] * labels.size(0), [None] * labels.size(0), [None] * labels.size(0)
        new_lengths1, new_lengths2, new_lengths3 = [None] * labels.size(0), [None] * labels.size(0), [None] * labels.size(0)
        for i, j in mk_locate:

            # another with marker
            if j == 2:
                new_l1 = labels[i][r2f]
            elif j == 3:
                new_l1 = labels[i][f2r]
            else:
                print('error')
            new_labels1[i] = new_l1
            new_lengths1[i] = valid_len[i]

            # 2 no marker
            lab = labels[i].tolist()
            new_l2 = lab
            new_l2.pop(j)
            new_l_len = len(new_l2) - 1
            new_l2.extend([pad_id]*(padded_len - (new_l_len + 1)))
            new_labels2[i] = torch.tensor(new_l2)
            new_lengths2[i] = new_l_len

            new_l3 = torch.tensor(new_l2)[fnm2rnm]
            new_labels3[i] = torch.tensor(new_l3)
            new_lengths3[i] = new_l_len


        for i in not_has_mk:
            # another no marker
            new_l1 = labels[i][fnm2rnm]
            new_labels1[i] = new_l1
            new_lengths1[i] = valid_len[i]

            # 2 type with marker
            lab = labels[i].tolist()

            mk_location = 3
            new_l2 = lab.copy()
            new_l2.insert(mk_location, mk_idx)
            new_l2 = new_l2[:padded_len]
            new_l_len = len(new_l2) - 1
            new_labels2[i] = torch.tensor(new_l2)
            new_lengths2[i] = new_l_len

            mk_location = 2
            lab = labels[i].tolist()
            new_l3 = lab.copy()
            new_l3.insert(mk_location, mk_idx)
            new_l3 = new_l3[:padded_len]
            new_l_len = len(new_l3) - 1
            new_labels3[i] = torch.tensor(new_l3)
            new_lengths3[i] = new_l_len

        new_labels1 = torch.stack(new_labels1)
        new_labels2 = torch.stack(new_labels2)
        new_labels3 = torch.stack(new_labels3)

        new_lengths1 = torch.tensor(np.array(new_lengths1))
        new_lengths2 = torch.tensor(np.array(new_lengths2))
        new_lengths3 = torch.tensor(np.array(new_lengths3))

        corr2 = vari_len_acc_compute(new_labels1, message, new_lengths1)
        corr3 = vari_len_acc_compute(new_labels2, message, new_lengths2)
        corr4 = vari_len_acc_compute(new_labels3, message, new_lengths3)
        # print(f'corr2: {corr2.sum()}')
        # print(f'corr3: {corr3.sum()}')
        # print(f'corr4: {corr4.sum()}')

        t1 = torch.logical_or(corr3, corr4)
        t2 = torch.logical_or(corr1, corr2)

        acc = torch.logical_or(t1, t2) * 1.0

    return acc


class MyLoss_spk:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_inpute: Dict[str, torch.Tensor],
        is_training: True,
        eos_id,
        pad_id,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # cut length for computing loss: cutting_length(message, receiver_output, labels)
        message_, receiver_output_ = cutting_length(message, receiver_output, labels)

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        # count eos, not count sos
        valid_len = (labels == eos_id).nonzero()[:, -1]
        if is_training:
            corr = vari_len_acc_compute(labels, message_, valid_len)
            acc = corr * 1.0
        else:
            acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        for step, step_output in enumerate(receiver_output_):
            l = self.criterion(step_output.contiguous().view(
                batch_size, -1), labels[:, step + 1])
            accumulate_loss += l
        return accumulate_loss, {"acc": acc}


class MyLoss_spk_v2:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_inpute: Dict[str, torch.Tensor],
        is_training: True,
        eos_id,
        pad_id,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # cut length for computing loss: cutting_length(message, receiver_output, labels)
        message_, receiver_output_ = cutting_length(message, receiver_output, labels)

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        # count eos, not count sos
        valid_len = (labels == eos_id).nonzero()[:, -1]
        # if is_training:
        #     corr = vari_len_acc_compute(labels, message_, valid_len)
        #     acc = corr * 1.0
        # else:
        #     acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        corr = vari_len_acc_compute(labels, message_, valid_len)
        acc = corr * 1.0
        multi_acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        for step, step_output in enumerate(receiver_output_):
            l = self.criterion(step_output.contiguous().view(
                batch_size, -1), labels[:, step + 1])
            accumulate_loss += l
        return accumulate_loss, {"acc": acc, 'multi_acc':multi_acc}