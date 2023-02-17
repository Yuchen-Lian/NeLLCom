# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, './pytorch-seq2seq/')

import os
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
sys.path.insert(1, str(ROOT_DIR))

from datetime import datetime, timedelta
from typing import List

import pandas as pd
from itertools import product

import torch

import egg.core as core
from egg.zoo.my_gru_all60.data import get_dataloader, MyDataset, my_selected_split
from egg.zoo.my_gru_all60.game_callbacks import get_callbacks, v2_get_callbacks, v3_get_callbacks_no_earlystop
from egg.zoo.my_gru_all60.games import build_game_after_supervised
from egg.zoo.my_gru_all60.games_comm import build_game_comm_spk, build_game_comm_lst, v2_build_game_comm_spk
from egg.zoo.my_gru_all60.utils import get_opts

import wandb

# wandb.init(project="my-test-project", entity="ylian")


def main(params: List[str], suffix) -> None:
    begin = datetime.now() + timedelta(hours=9)
    print(f"| STARTED JOB at {begin}...")

    opts = get_opts(params=params)
    # opts.n_epochs = 10
    opts.n_epochs = 60
    opts.num_workers = 0
    # opts.meaning_embedding_dim = 8
    opts.dump_every = 10
    opts.lr = 0.01
    # opts.speaker_hidden_size = 128
    # opts.listener_hidden_size = 128
    opts.do_padding = True # standard with padding

    opts.rnn = 'gru'
    # opts.patience = 5
    # patience = 5 for fix_op
    # opts.patience = 10
    # patience = 10 for free_op

    opts.patience = 10 if 'free' in opts.language else 5

    print(f"{opts}\n")
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    dataset_dir = os.path.join(opts.dataset_folder, opts.language, opts.dataset_filename)

    full_dataset = MyDataset(dataset_dir, opts.language)

    training_data_size = opts.trainset_proportion

    language = opts.language

    train_size = int(training_data_size * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # train_d, test_d = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_d, test_d = my_selected_split(full_dataset, [train_size, test_size], selected=True)
    # selected=True:    all elements appear in the train set,
    #                   train_size MUST sent first!!!

    train_loader_test = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )
    test_loader_test = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
        drop_last=False,
        shuffle=False,
    )

    log_dir_lst = opts.log_dir.split('.')[0] + '_' + suffix + '_lst.txt'
    outputs_dir_lst = os.path.join(opts.outputs_dir, suffix+'_lst')
    if not os.path.exists(outputs_dir_lst):
        os.mkdir(outputs_dir_lst)
    dump_dir_lst = os.path.join(opts.dump_dir, suffix+'_lst')
    if not os.path.exists(dump_dir_lst):
        os.mkdir(dump_dir_lst)
    save_model_dir = os.path.join(opts.save_model_dir, suffix)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    game_lst = build_game_comm_lst(
        train_data=full_dataset,
        encoder_hidden_size=opts.listener_hidden_size,
        is_distributed=opts.distributed_context.is_distributed,
        rnn_cell=opts.rnn
    )

    optimizer = core.build_optimizer(game_lst.parameters())
    optimizer_scheduler = None

    # callbacks = get_callbacks(log_dir=log_dir_lst, acc_threshhold=0.999, dump_output=opts.dump_output, outputs_dir=outputs_dir_lst, dump_every=opts.dump_every, save_model_dir=save_model_dir)
    callbacks = v3_get_callbacks_no_earlystop(log_dir=log_dir_lst, acc_threshhold=0.999, patience=opts.patience, dump_output=opts.dump_output, outputs_dir=outputs_dir_lst, dump_every=opts.dump_every, save_model_dir=save_model_dir)

    trainer_lst = core.Trainer(
        game=game_lst,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader_test,
        validation_data=test_loader_test,
        callbacks=callbacks,
    )
    trainer_lst.train(n_epochs=opts.n_epochs)

    print(f"| FINISHED supervised listening training")

    if opts.dump_output:
        arrange_dump_v2(dataset=full_dataset, output_dir=outputs_dir_lst, dump_dir=dump_dir_lst, mode='lst')

    # ================================================================

    # game_spk = build_game_comm_spk(
    #     train_data=full_dataset,
    #     meaning_embedding_size=opts.meaning_embedding_dim,
    #     decoder_hidden_size=opts.speaker_hidden_size,
    #     is_distributed=opts.distributed_context.is_distributed,
    #     rnn_cell=opts.rnn,
    #     spk_max_len=opts.spk_max_len
    # )

    game_spk = v2_build_game_comm_spk(
        train_data=full_dataset,
        meaning_embedding_size=opts.meaning_embedding_dim,
        decoder_hidden_size=opts.speaker_hidden_size,
        is_distributed=opts.distributed_context.is_distributed,
        rnn_cell=opts.rnn,
        spk_max_len=opts.spk_max_len
    )

    optimizer = core.build_optimizer(game_spk.parameters())
    optimizer_scheduler = None


    log_dir_spk = opts.log_dir.split('.')[0] + '_' + suffix + '_spk.txt'
    outputs_dir_spk = os.path.join(opts.outputs_dir, suffix+'_spk')
    if not os.path.exists(outputs_dir_spk):
        os.mkdir(outputs_dir_spk)
    dump_dir_spk = os.path.join(opts.dump_dir, suffix+'_spk')
    if not os.path.exists(dump_dir_spk):
        os.mkdir(dump_dir_spk)
    save_model_dir = os.path.join(opts.save_model_dir, suffix)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    # callbacks = get_callbacks(log_dir=log_dir_spk, acc_threshhold=0.999, dump_output=opts.dump_output, outputs_dir=outputs_dir_spk, dump_every=opts.dump_every, save_model_dir=save_model_dir)
    callbacks = v3_get_callbacks_no_earlystop(log_dir=log_dir_spk, acc_threshhold=0.999, patience=5, dump_output=opts.dump_output, outputs_dir=outputs_dir_spk, dump_every=opts.dump_every, save_model_dir=save_model_dir)


    trainer_spk = core.Trainer(
        game=game_spk,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader_test,
        validation_data=test_loader_test,
        callbacks=callbacks,
    )
    trainer_spk.train(n_epochs=opts.n_epochs)

    print(f"| FINISHED supervised speaking training")

    if opts.dump_output:
        arrange_dump_v2(dataset=full_dataset, output_dir=outputs_dir_spk, dump_dir=dump_dir_spk, mode='spk')

    # ================================================================

    my_trained_speaker = trainer_spk.game.model
    my_trained_listener = trainer_lst.game.model

    # self play
    game_selfplay = build_game_after_supervised(
        opts,
        speaker=my_trained_speaker,
        listener=my_trained_listener,
        train_data=full_dataset,
        meaning_embedding_size=opts.meaning_embedding_dim,
        encoder_hidden_size=opts.listener_hidden_size,
        decoder_hidden_size=opts.speaker_hidden_size,
        is_distributed=opts.distributed_context.is_distributed,
        game_type='commu'
    )

    optimizer = core.build_optimizer(game_selfplay.parameters())
    optimizer_scheduler = None

    log_dir_comm = opts.log_dir.split('.')[0] + '_' + suffix + '_comm.txt'
    outputs_dir_comm = os.path.join(opts.outputs_dir, suffix+'_comm')
    if not os.path.exists(outputs_dir_comm):
        os.mkdir(outputs_dir_comm)
    dump_dir_comm = os.path.join(opts.dump_dir, suffix+'_comm')
    if not os.path.exists(dump_dir_comm):
        os.mkdir(dump_dir_comm)
    save_model_dir = os.path.join(opts.save_model_dir, suffix)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    # callbacks = get_callbacks(log_dir=log_dir_comm, acc_threshhold=0.999, dump_output=opts.dump_output, outputs_dir=outputs_dir_comm, dump_every=opts.dump_every, save_model_dir=save_model_dir)
    callbacks = v3_get_callbacks_no_earlystop(log_dir=log_dir_comm, acc_threshhold=0.999, patience=5, dump_output=opts.dump_output, outputs_dir=outputs_dir_comm, dump_every=opts.dump_every, save_model_dir=save_model_dir)

    trainer_selfplay = core.Trainer(
        game=game_selfplay,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader_test,
        validation_data=test_loader_test,
        callbacks=callbacks,
    )
    trainer_selfplay.train(n_epochs=opts.n_epochs)

    if opts.dump_output:
        arrange_dump_v2(dataset=full_dataset, output_dir=outputs_dir_comm, dump_dir=dump_dir_comm, mode='comm')

    end = datetime.now() + timedelta(hours=9)  # Using CET timezone

    print(f"| FINISHED JOB at {end}. It took {end - begin}")
    print(f"Language: {language}")
    print(f"trainset_size: {train_size}, testset_size: {test_size}")
    print(f"batch_size: {opts.batch_size}, emb_dim: {opts.meaning_embedding_dim}, hidden_size: {opts.speaker_hidden_size}")


def multi_run(hyperdict, rename, test_suffix=''):
    grid = list(product(*hyperdict.values()))
    multi_params = []
    suffix = []
    for value_set in grid:
        v_list = list(zip(hyperdict.keys(), value_set))
        v_str = [value2params(v) for v in v_list]
        multi_params.append(v_str)
        v_rename = '_'.join([value2str(v, rename) for v in v_list])
        # just_test
        v_rename = test_suffix + v_rename
        suffix.append(v_rename)

    return multi_params, suffix


def value2params(pair):
    para, v = pair
    return '--'+ para + '=' + str(v)


def value2str(pair, name):
    para, v = pair
    return name[para] + str(v)


def substr_rindex(s, subs):
    if subs in s:
        ind = s.index(subs) + len(subs)
        if s[ind:].find('_') == -1:
            next_ = None
        else:
            next_ = s[ind:].index('_')+ind
        return s.index(subs) + len(subs), next_
    else:
        return False


def myloading_tensor(dir, f_list):
    tensor_dict = {}
    if len(f_list) > 0:
        for f in f_list:
            t = torch.load(os.path.join(dir, f))
            x, y = substr_rindex(f, 'epoch')
            epoch = int(f[x:y])
            tensor_dict[epoch] = t
    return tensor_dict


def arrange_dump_spk(dataset, output_dir, dump_dir):
    files = os.listdir(output_dir)
    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_label = [f for f in files if 'label' in f]
    label_dict = myloading_tensor(output_dir, f_label)

    f_input = [f for f in files if 'input' in f]
    input_dict = myloading_tensor(output_dir, f_input)

    if msg_dict.keys() == label_dict.keys() == input_dict.keys():
        for k in msg_dict.keys():
            msg = msg_dict[k]
            lab = label_dict[k]
            inp = input_dict[k]
            total_steps = lab.shape[1]-1

            if not(msg.size(0) == lab.size(0) == inp.size(0)):
                break

            msg_token, lab_token, inp_token = [], [], []
            for i in range(msg.size(0)):
                msg_t = ' '.join(dataset.vocab_utterance.lookup_tokens(msg[i].tolist()))
                lab_t = ' '.join(dataset.vocab_utterance.lookup_tokens(lab[i].tolist()))
                inp_t = ' '.join(dataset.vocab_meaning.lookup_tokens(inp[i].tolist()))
                msg_token.append(msg_t)
                lab_token.append(lab_t)
                inp_token.append(inp_t)

            df = pd.DataFrame({'input': inp_token, 'label':lab_token, 'message': msg_token})
            df.to_csv(os.path.join(dump_dir, f'dump_epoch{k}.txt'), sep='\t')
    return


def arrange_dump_lst(dataset, output_dir, dump_dir):
    files = os.listdir(output_dir)
    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_label = [f for f in files if 'label' in f]
    label_dict = myloading_tensor(output_dir, f_label)

    f_input = [f for f in files if 'input' in f]
    input_dict = myloading_tensor(output_dir, f_input)

    if msg_dict.keys() == label_dict.keys() == input_dict.keys():
        for k in msg_dict.keys():
            msg = msg_dict[k]
            lab = label_dict[k]
            inp = input_dict[k]
            total_steps = lab.shape[1]-1

            if not(msg.size(0) == lab.size(0) == inp.size(0)):
                break

            msg_token, lab_token, inp_token = [], [], []
            for i in range(msg.size(0)):
                msg_t = ' '.join(dataset.vocab_meaning.lookup_tokens(msg[i].tolist()))
                lab_t = ' '.join(dataset.vocab_meaning.lookup_tokens(lab[i].tolist()))
                inp_t = ' '.join(dataset.vocab_utterance.lookup_tokens(inp[i].tolist()))
                msg_token.append(msg_t)
                lab_token.append(lab_t)
                inp_token.append(inp_t)

            df = pd.DataFrame({'uttr': inp_token, 'meaning':lab_token, 'guess': msg_token})
            df.to_csv(os.path.join(dump_dir, f'dump_epoch{k}.txt'), sep='\t')
    return

def arrange_dump_v2(dataset, output_dir, dump_dir, mode):
    files = os.listdir(output_dir)

    f_uttr = [f for f in files if 'uttr' in f]
    uttr_dict = myloading_tensor(output_dir, f_uttr)

    f_mean = [f for f in files if 'mean' in f]
    mean_dict = myloading_tensor(output_dir, f_mean)

    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_lstpred = [f for f in files if 'lstpred' in f]
    lstpred_dict = myloading_tensor(output_dir, f_lstpred)

    if uttr_dict.keys() == mean_dict.keys() == msg_dict.keys() == lstpred_dict.keys():
        for k in msg_dict.keys():
            uttr = uttr_dict[k]
            mean = mean_dict[k]
            msg = msg_dict[k]
            lstpred = lstpred_dict[k]
            total_steps = uttr.shape[1]-1

            if not(uttr.size(0) == mean.size(0)):
                break

            msg_token, uttr_token, mean_token, lstpred_token = [], [], [], []
            for i in range(uttr.size(0)):
                uttr_t = ' '.join(dataset.vocab_utterance.lookup_tokens(uttr[i].tolist()))
                mean_t = ' '.join(dataset.vocab_meaning.lookup_tokens(mean[i].tolist()))
                msg_t = ' '.join(dataset.vocab_utterance.lookup_tokens(msg[i].tolist())) if mode in ['comm', 'spk'] else ''
                lstpred_t = ' '.join(dataset.vocab_meaning.lookup_tokens(lstpred[i].tolist())) if mode in ['comm', 'lst'] else ''
                msg_token.append(msg_t)
                uttr_token.append(uttr_t)
                mean_token.append(mean_t)
                lstpred_token.append(lstpred_t)

            df = pd.DataFrame({'meaning': mean_token, 'utterance': uttr_token,
                               'message': msg_token, 'listener_prediction': lstpred_token})
            df.to_csv(os.path.join(dump_dir, f'dump_epoch{k}.txt'), sep='\t')
    return

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    # 'free_op'
    # 'fix_op'

    hyperdict = {'language': ['fix_op', 'free_op'], 'trainset_proportion': [0.667], 'speaker_hidden_size': [128],
                 'meaning_embedding_dim': [8], 'batch_size': [32], 'random_seed': [2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239]}
                 # ['fix_mk2', 'fix_op', 'free_mk', 'free_op']
                 # [2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239]
                # [1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239]

    rename = {'language': 'lang', 'trainset_proportion': 'split', 'speaker_hidden_size': 'hidden',
              'meaning_embedding_dim': 'emb', 'batch_size': 'batch', 'random_seed': 'seed'}

    multi_params, suffix = multi_run(hyperdict, rename, test_suffix='')

    for param_set, suf in zip(multi_params, suffix):
        main(param_set, suf)




