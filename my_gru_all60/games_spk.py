# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from egg.core.interaction import LoggingStrategy
from egg.zoo.my_gru_all60.archs_spk import Speaker_Game, Speaker_encoder, Speaker_decoder, Speaker
from egg.zoo.my_gru_all60.losses_spk import MyLoss_spk


def build_game_spk(
        train_data,
        meaning_embedding_size: int = 32,
        decoder_hidden_size: int = 128,
        is_distributed: bool = False,
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    # loss = MyLoss(nn.NLLLoss())

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    sos_id, eos_id, pad_id = train_data.get_special_index()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    uttr_max_len = 10

    loss = MyLoss_spk(nn.NLLLoss(ignore_index=pad_id))

    speaker_enc = Speaker_encoder(vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size, max_len=meaning_max_len, output_size=decoder_hidden_size)
    speaker_dec = Speaker_decoder(vocab_size=uttr_vocab_size, max_len=uttr_max_len, hidden_size=decoder_hidden_size, sos_id=sos_id, eos_id=eos_id, rnn_cell='gru', use_attention=False, pad_id=pad_id)
    speaker = Speaker(speaker_enc, speaker_dec)
    game = Speaker_Game(speaker, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game