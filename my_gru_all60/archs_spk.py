# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.interaction import LoggingStrategy

import numpy as np
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from seq2seq.models import DecoderRNN


class Speaker(nn.Module):
    def __init__(self, spk_enc, spk_dec):
        super(Speaker, self).__init__()
        self.name = 'Speaker'
        self.encoder = spk_enc
        self.decoder = spk_dec
        self.eos_id = spk_dec.eos_id
        self.sos_id = spk_dec.sos_id
        self.pad_id = spk_dec.pad_id

    def forward(
            self,
            sender_input: torch.Tensor,
            labels: torch.Tensor,
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:
        prev_hidden = self.encoder(sender_input, aux_input)
        # if receiver_input is None:
        receiver_input = labels
        receiver_output, message, message_length, entropy = self.decoder(encoder_hidden=torch.unsqueeze(prev_hidden, 0),
                                                                         inputs=receiver_input, eval=not self.training, teacher_forcing_ratio=1)
        return receiver_output, message, message_length, entropy


class Speaker_Game(nn.Module):
    def __init__(
            self,
            speaker: nn.Module,
            loss: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, Dict[str, Any]],
            ],
            train_logging_strategy: Optional[LoggingStrategy] = None,
            test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(Speaker_Game, self).__init__()
        self.model = speaker
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
            sender_input: torch.Tensor,
            labels: torch.Tensor,
            receiver_input: torch.Tensor = None,
            aux_input=None,
    ) -> Tuple[torch.Tensor, Interaction]:
        # if receiver_input is None:
        receiver_input = labels
        receiver_output, message, message_length, entropy = self.model(sender_input, labels=labels,
                                                                       receiver_input=receiver_input,
                                                                       aux_input=aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input, self.training,
            eos_id=self.model.eos_id, pad_id=self.model.pad_id
        )
        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        receiver_output_tensor = torch.stack(receiver_output).permute(1, 0, -1).squeeze()
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input, # mean
            receiver_input=None,
            labels=labels, # uttr
            aux_input=aux_input,
            receiver_output=None,
            message=message, # msg (speaker prediction)
            message_length=message_length,
            aux=aux_info,
        )
        return loss.mean(), interaction


class Speaker_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, output_size, activation_fn=None):
        super(Speaker_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.requires_grad = True
        in_features = embedding_size * max_len
        self.activation_fn = activation_fn
        self.fc_out = nn.Linear(in_features, output_size)

    def forward(
            self,
            sender_input: torch.Tensor,
            aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        batch_size = sender_input.size(0)
        embedded = self.embedding(sender_input)
        embedded_concat = embedded.view(batch_size, -1)
        # context = self.activation_fn(self.fc_out(embedded_concat))
        context = self.fc_out(embedded_concat)
        return context


class Speaker_decoder(DecoderRNN):
    """
    Drop-in replacement for DecoderRNN that _always_ samples sequences (even during the evaluation phase).
    """

    def __init__(self, pad_id, *args, **kwargs):
        super(Speaker_decoder, self).__init__(*args, **kwargs)
        self.pad_id = pad_id

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

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            # NB: here we changed argmax for sampling
            if eval == True:
                symbols = decoder_outputs[-1].topk(1)[1]
            else:
                symbols = Categorical(logits=step_output).sample().unsqueeze(1)
            #
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
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
                symbols = decode(di, step_output)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        message = torch.stack(sequence_symbols).permute(1, 0, -1).squeeze()
        msg_lengths = torch.tensor(lengths)

        return decoder_outputs, message, msg_lengths
