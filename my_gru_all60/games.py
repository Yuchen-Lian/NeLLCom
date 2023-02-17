# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from egg.core.interaction import LoggingStrategy
from egg.zoo.my_gru_all60.archs import Commu_Game2, SpeakerListener
from egg.zoo.my_gru_all60.losses import MyLoss
from egg.zoo.my_gru_all60.games_lst import build_game_lst
from egg.zoo.my_gru_all60.games_spk import build_game_spk
from egg.zoo.my_gru_all60.data import MyDataset


def build_game_after_supervised(
        opts,
        speaker,
        listener,
        train_data: MyDataset,
        meaning_embedding_size: int = 32,
        encoder_hidden_size: int = 128,
        decoder_hidden_size: int = 128,
        is_distributed: bool = False,
        game_type: str = 'commu',
) -> nn.Module:

    if game_type == 'speak':
        game = build_game_spk(
            dataset=train_data,
            encoder_hidden_size=opts.speaker_hidden_size,
            is_distributed=opts.distributed_context.is_distributed,
        )
        train_mode = 'supervised'
    elif game_type == 'listen':
        game = build_game_lst(
            dataset=train_data,
            encoder_hidden_size=opts.listener_hidden_size,
            is_distributed=opts.distributed_context.is_distributed,
        )
        train_mode = 'supervised'
    elif game_type == 'commu':
        train_mode = 'reinforce'

        # load_speaker
        # load_listener

        train_logging_strategy = LoggingStrategy.minimal()
        train_logging_strategy.store_aux_input =True

        loss = MyLoss(nn.NLLLoss())

        spk_lst = SpeakerListener(speaker, listener, do_padding=opts.do_padding)
        game2 = Commu_Game2(train_mode=train_mode, spk_lst=spk_lst, loss=loss, train_logging_strategy=train_logging_strategy)

        if is_distributed:
            game2 = nn.SyncBatchNorm.convert_sync_batchnorm(game2)

        return game2



