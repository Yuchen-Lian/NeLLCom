# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union, Optional
from collections import defaultdict

import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.interaction import LoggingStrategy
from egg.core.baselines import Baseline, MeanBaseline

import numpy as np
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from seq2seq.models import DecoderRNN


class SpeakerListener(nn.Module):
    def __init__(
            self,
            speaker: nn.Module,
            listener: nn.Module,
            do_padding=True
    ):
        super(SpeakerListener, self).__init__()
        self.speaker = speaker
        self.listener = listener
        self.name = 'Spk_Lst'
        self.eos_id = speaker.eos_id
        self.sos_id = speaker.sos_id
        self.pad_id = speaker.pad_id
        self.do_padding = do_padding

    def forward(
            self,
            sender_input: torch.Tensor,
            labels: torch.Tensor,
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:
        # # SPEAKING
        speaker_output, message, message_length, entropy_spk = self.speaker(sender_input, labels=labels,
                                                                            receiver_input=receiver_input,
                                                                            aux_input=aux_input)
        speaker_output = torch.stack(speaker_output).permute(1, 0)

        # # attach [sos] to message
        # sos_ = torch.stack([torch.tensor(self.speaker.decoder.sos_id)] * message.size(0))
        # message = torch.cat((sos_.unsqueeze(-1), message), dim=1)
        padded_message, message_length = my_padding(message, self.speaker.eos_id, self.speaker.pad_id, do_padding=self.do_padding)

        # # LISTENING
        # # switch input&lable
        listener_output, listener_prediction, logits_lst, entropy_lst = self.listener(sender_input=padded_message,
                                                                                      labels=sender_input,
                                                                                      receiver_input=None,
                                                                                      aux_input=None)
        return speaker_output, padded_message, message_length, entropy_spk, listener_output, listener_prediction, logits_lst, entropy_lst


class Commu_Game2(nn.Module):
    def __init__(
            self,
            train_mode: str,
            spk_lst: nn.Module,
            loss: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, Dict[str, Any]],
            ],
            train_logging_strategy: Optional[LoggingStrategy] = None,
            test_logging_strategy: Optional[LoggingStrategy] = None,
            spk_entropy_coeff: float = 0.1,
            lst_entropy_coeff: float = 0,
            length_cost: float = 0.0,
            baseline_type: Baseline = MeanBaseline,
    ):
        super(Commu_Game2, self).__init__()
        self.train_mode = train_mode
        self.model = spk_lst
        self.loss = loss

        self.spk_entropy_coeff = spk_entropy_coeff
        self.lst_entropy_coeff = lst_entropy_coeff
        self.length_cost = length_cost

        self.baselines = defaultdict(baseline_type)

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

        self.model.speaker.decoder.set_train_mode(self.train_mode)

    def forward(
            self,
            sender_input: torch.Tensor,
            labels: torch.Tensor,
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:

        receiver_input = receiver_input.items() if any(receiver_input) else None
        speaker_output, message, message_length, entropy_spk, listener_output, listener_prediction, logits_lst, entropy_lst = self.model(
            sender_input, labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input)

        loss, aux_info = self.loss(
            sender_input,
            sender_input,
            speaker_output,
            message,
            listener_output,
            listener_prediction,
            aux_input,
            self.training)

        effective_entropy_spk = torch.zeros_like(entropy_lst)
        effective_log_prob_spk = torch.zeros_like(logits_lst)

        for i in range(message.size(1)):
            not_eosed = (i < message_length).float()
            effective_entropy_spk += entropy_spk[:, i] * not_eosed
            effective_log_prob_spk += speaker_output[:, i] * not_eosed
        effective_entropy_spk = effective_entropy_spk / message_length.float()

        weighted_entropy = (
                effective_entropy_spk.mean() * self.spk_entropy_coeff
                + entropy_lst.mean() * self.lst_entropy_coeff
        )

        log_prob = effective_log_prob_spk + logits_lst

        length_loss = message_length.float() * self.length_cost

        policy_length_loss = (
                (length_loss - self.baselines["length"].predict(length_loss))
                * effective_log_prob_spk
        ).mean()
        policy_loss = (
                (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        ).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines["loss"].update(loss)
            self.baselines["length"].update(length_loss)

        aux_info["spk_entropy"] = entropy_spk.detach()
        aux_info["lst_entropy"] = entropy_lst.detach()
        aux_info["length"] = message_length.float()  # will be averaged

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=listener_prediction,
            message=message,
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )
        return optimized_loss, interaction


class RLSpeaker_decoder(DecoderRNN):
    def __init__(self, pad_id, *args, **kwargs):
        super(RLSpeaker_decoder, self).__init__(*args, **kwargs)
        self.pad_id = pad_id

    def set_train_mode(self, train_mode='supervised'):
        if train_mode == 'supervised':
            self.rl = False
        elif train_mode == 'reinforce':
            self.rl = True
        else:
            self.rl = False
        return

    def forward(self, inputs, encoder_hidden, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=1, eval=False):
        ret_dict = dict()
        if self.use_attention or encoder_outputs is not None:
            raise ValueError("Nothing to attend in this scenario, No encoder_output")

        if eval:
            inputs = None
            teacher_forcing_ratio = 0

        if self.rnn_cell == nn.LSTM:
            c_zeros = torch.zeros_like(encoder_hidden)
            encoder_hidden = (encoder_hidden, c_zeros)

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if self.rl:
            use_teacher_forcing = False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        entropy = []

        def decode(step, step_output):
            if self.rl:
                distr = Categorical(logits=step_output)
                entropy.append(distr.entropy())
                if eval:
                    symbols = step_output.argmax(dim=1)
                else:
                    symbols = distr.sample()
                decoder_outputs.append(distr.log_prob(symbols))
                sequence_symbols.append(symbols.unsqueeze(1))
            else:
                # NB: here we changed argmax for sampling
                decoder_outputs.append(step_output)
                if eval == True:
                    symbols = decoder_outputs[-1].topk(1)[1]
                else:
                    symbols = Categorical(logits=step_output).sample().unsqueeze(1)
                sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

            if symbols.dim() < 2:
                symbols = symbols.unsqueeze(1)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, _ = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                      function)
                step_output = decoder_output.squeeze(1)
                # symbols = decode(di, step_output)
                # decoder_input = symbols.unsqueeze(1)
                decoder_input = decode(di, step_output)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        message = torch.stack(sequence_symbols).permute(1, 0, -1).squeeze()
        msg_lengths = torch.tensor(lengths)
        entropy = torch.stack(entropy).permute(1, 0) if len(entropy)>0 else torch.zeros_like(message)

        # NOT YET!!!
        # zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
        #
        # sequence = torch.cat([sequence, zeros.long()], dim=1)
        # logits = torch.cat([logits, zeros], dim=1)
        # entropy = torch.cat([entropy, zeros], dim=1)

        return decoder_outputs, message, msg_lengths, entropy


def my_padding(messages: torch.Tensor, eos_id, pad_id, do_padding:True) -> torch.Tensor:
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).

    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)

    if do_padding:
        zero_mask = messages == eos_id
        # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
        # querying torch.nonzero()
        # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
        # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
        # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
        # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps  happened after 0 occured (including it)
        # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

        lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
        lengths.add_(1).clamp_(max=max_k)

        for i, j in zero_mask.nonzero():
            messages[i, j + 1:] = torch.tensor(pad_id)
    else:
        lengths = torch.full((messages.size(0),), messages.size(1))


    return messages, lengths
