# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from egg.core.interaction import LoggingStrategy
from egg.zoo.my_gru_all60.archs_lst import Listener_Game, Listener_encoder, Listener_decoder, Listener
from egg.zoo.my_gru_all60.losses_lst import MyLoss_lst


def build_game_lst(
        dataset,
        encoder_hidden_size: int = 128,
        is_distributed: bool = False,
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    loss = MyLoss_lst(nn.NLLLoss())

    meaning_vocab_size, uttr_vocab_size = dataset.get_vocab_size()
    meaning_max_len, uttr_max_len = dataset.get_max_len()
    sos_id, eos_id, pad_id = dataset.get_special_index()

    listener_enc = Listener_encoder(vocab_size=uttr_vocab_size, max_len=meaning_max_len, hidden_size=encoder_hidden_size, rnn_cell='gru', variable_lengths=True, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id)
    listener_dec = Listener_decoder(vocab_size=meaning_vocab_size, meaning_len=meaning_max_len, input_size=encoder_hidden_size)
    listener = Listener(listener_enc, listener_dec)
    game = Listener_Game(listener, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
