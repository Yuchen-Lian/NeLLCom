# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.interaction import LoggingStrategy

from seq2seq.models import DecoderRNN, EncoderRNN


class Listener_decoder(nn.Module):
    def __init__(self, vocab_size, meaning_len, input_size, activation_fn=nn.ReLU()):
        super(Listener_decoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding.weight.requires_grad = True
        self.meaning_len = meaning_len
        out_features = vocab_size * meaning_len
        self.linear = nn.Linear(input_size, out_features)
        self.activation_fn = activation_fn
        self.out_logits = nn.LogSoftmax(dim=-1)


    def forward(
            self,
            receiver_input: torch.Tensor,
            aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        batch_size = receiver_input.size(0)
        # output = self.activation_fn(self.linear(receiver_input))
        output = self.linear(receiver_input)
        output = self.out_logits(output.view(batch_size, self.meaning_len, -1))
        prediction = output.argmax(dim=-1)
        return output, prediction


class Listener_encoder(EncoderRNN):
    def __init__(self, sos_id, eos_id, pad_id, *args, **kwargs):
        super(Listener_encoder, self).__init__(*args, **kwargs)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id


    def forward(self, input_var, input_lengths=None):
        return super().forward(input_var, input_lengths)


class Listener(nn.Module):
    def __init__(self, lst_enc, lst_dec):
        super(Listener, self).__init__()
        self.name = 'Listener'
        self.encoder = lst_enc
        self.decoder = lst_dec
        self.eos_id = lst_enc.eos_id
        self.sos_id = lst_enc.sos_id
        self.pad_id = lst_enc.pad_id

    def forward(
            self,
            sender_input: torch.Tensor, # should be utterances
            labels: torch.Tensor, # should be meaning
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:

        input_length = self.get_length(sender_input)

        _, uttr_representation = self.encoder(sender_input, input_length)
        if self.encoder.rnn_cell == nn.LSTM:
            uttr_representation = uttr_representation[0]

        uttr_representation = uttr_representation.squeeze()
        listener_output, prediction = self.decoder(uttr_representation)

        logits = torch.zeros(listener_output.size(0)).to(listener_output.device)
        entropy = logits

        return listener_output, prediction, logits, entropy

    def get_length(self, input):
        eos_id = self.eos_id
        max_k = input.size(1)
        zero_mask = input == eos_id
        lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
        lengths.add_(1).clamp_(max=max_k)
        return lengths



class Listener_Game(nn.Module):
    def __init__(
            self,
            listener: nn.Module,
            loss: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, Dict[str, Any]],
            ],
            train_logging_strategy: Optional[LoggingStrategy] = None,
            test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(Listener_Game, self).__init__()
        self.model = listener
        self.loss = loss

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

    def forward(
            self,
            original_sender_input: torch.Tensor, # should be utterances
            original_labels: torch.Tensor, # should be meaning
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:

        # switch input&lable
        sender_input = original_labels
        labels = original_sender_input
        receiver_input = receiver_input.items() if any(receiver_input) else None

        listener_output, prediction, logits, entropy = self.model(sender_input=sender_input, labels=labels, receiver_input=receiver_input, aux_input=None,)

        loss, aux_info = self.loss(
            sender_input=sender_input,
            message=prediction,
            receiver_input=receiver_input,
            receiver_output=listener_output,
            labels=labels,
            aux_input=aux_input,
        )
        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=original_sender_input, # mean
            receiver_input=None,
            labels=original_labels, # uttr
            aux_input=aux_input,
            receiver_output=prediction, # listener prediction
            message=None,
            message_length=None,
            aux=aux_info,
        )
        return loss.mean(), interaction


