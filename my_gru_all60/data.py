# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Iterable, Optional, Tuple
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchtext.vocab import build_vocab_from_iterator


def get_dataloader(
        train_dataset,
        batch_size: int = 32,
        num_workers: int = 1,
        is_distributed: bool = False,
        seed: int = 111,
        drop_last: bool = True,
        shuffle: bool = True,
) -> Iterable[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
]:
    "Returning an iterator for tuple(sender_input, labels, receiver_input)."

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return train_loader


class MyDataset(Dataset):
    """Meaning-Utterance dataset."""

    def __init__(self, csv_file, language):
        """
        :param csv_file: Path to the csv file
        """
        self.get_language_type(language)
        self.raw_samples = pd.read_csv(csv_file, sep='\t', names=['meaning', 'utterance'])
        self._init_dataset()
        self.max_meaning_len, self.max_uttr_len = self.get_max_len()

    def get_language_type(self, language):
        if 'fix_mk' in language:
            self.language = 'fix_mk'
        elif 'fix_op' in language:
            self.language = 'fix_op'
        elif 'fix' in language and '_' not in language:
            self.language = 'fix'
        elif 'free_mk' in language:
            self.language = 'free_mk'
        elif 'free_op' in language:
            self.language = 'free_op'
        elif 'free' in language and '_' not in language:
            self.language = 'free'
        else:
            self.language = 'None'
            print('Language type not defined')

    def __len__(self):
        return len(self.raw_samples)

    def sample_utterance_order(self, utterance_tensor):

        mk_idx = self.get_mk_index()
        mk_locate = torch.nonzero(torch.eq(utterance_tensor, mk_idx)).item()

        if random.random() < 0.5:

            f2r = [0, 2, 3, 1, 4, 5]
            r2f = [0, 3, 1, 2, 4, 5]

            if mk_locate == 2:
                new_l = utterance_tensor[r2f]
            elif mk_locate == 3:
                new_l = utterance_tensor[f2r]
            else:
                new_l = None
        else:
            new_l = utterance_tensor

        new_mk_locate = torch.nonzero(torch.eq(new_l, mk_idx)).item()
        if new_mk_locate == 2:
            order = -1
        elif new_mk_locate == 3:
            order = 1

        return new_l, order

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        meaning = self.samples['meaning'].iloc[idx]
        meaning_tensor = torch.tensor(self.vocab_meaning(meaning))

        utterance = self.samples['utterance'].iloc[idx]
        utterance_ = utterance + ['<PAD>']*(self.max_uttr_len - len(utterance))
        utterance_tensor = torch.tensor(self.vocab_utterance(utterance_))

        # # sample = {'meaning': meaning_tensor, 'utterance': utterance_tensor}
        # sample = [meaning_tensor, utterance_tensor]

        language_dict = {'fix': 0, 'fix_mk': 1, 'free_mk': 2, 'fix_op': 3, 'free_op': 4}
        lang_code = torch.tensor(language_dict[self.language])

        if self.language == 'free_mk':
            utterance_tensor, order = self.sample_utterance_order(utterance_tensor)
        else:
            order = 0

        sample = [meaning_tensor, utterance_tensor, {}, {'language': lang_code, 'mk_idx': self.get_mk_index(), 'order':order}]

        return sample

    def _init_dataset(self):
        # Preprocessing
        meaning_fn = lambda x: x.split()
        utterance_fn = lambda x: ['<SOS>'] + x.split() + ['<EOS>']

        self.samples = self.raw_samples.copy()
        self.samples['meaning'] = self.samples['meaning'].apply(meaning_fn)
        self.samples['utterance'] = self.samples['utterance'].apply(utterance_fn)

        # build vocab
        counter_meaning = Counter()
        counter_utterance = Counter()
        for ind, content in self.raw_samples.iterrows():
            m, u = content
            counter_meaning.update(m.split())
            counter_utterance.update(u.split())

        self.vocab_meaning = build_vocab_from_iterator(iter([[i] for i in counter_meaning]))
        self.vocab_utterance = build_vocab_from_iterator(iter([[i] for i in counter_utterance]),
                                                         specials=['<SOS>', '<EOS>', '<PAD>'])

    def get_special_index(self):
        return self.vocab_utterance(['<SOS>', '<EOS>', '<PAD>'])

    def get_vocab_size(self):
        return len(self.vocab_meaning), len(self.vocab_utterance)

    def get_max_len(self):
        max_l_meaning = max(self.samples['meaning'].apply(lambda x: len(x)))
        max_l_uttr = max(self.samples['utterance'].apply(lambda x: len(x)))
        return max_l_meaning, max_l_uttr

    def get_mk_index(self):
        if self.language in ['fix_op', 'fix_mk', 'free_op', 'free_mk']:
            return self.vocab_utterance['mk']
        else:
            return -1


from typing import (
    List,
    Optional,
    Sequence,
    TypeVar,
)

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, randperm
from torch._utils import _accumulate

from torch._C import Generator

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

UNTRACABLE_DATAFRAME_PIPES = ['batch',  # As it returns DataChunks
                              'groupby',   # As it returns DataChunks
                              '_dataframes_as_tuples',  # As it unpacks DF
                              'trace_as_dataframe',  # As it used to mark DF for tracing
                              ]

def my_selected_split(dataset: Dataset[T], lengths: Sequence[int],
                   generator: Optional[Generator] = default_generator, selected=False) -> List[Subset[T]]:
    r"""
    ************************
    If selected=True: every element appears in the train set
    ************************

    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()

    if selected:
        # ensure every element appears in the train set

        meaning_list = dataset.vocab_meaning.get_itos()
        meaning_samples = dataset.samples['meaning']

        # solution 3: check current selected indices if all item appears
        train_samples = dataset.samples.loc[indices[:480]]['meaning']

        def df_set(x, s):
            return s.update(x)
        s = set()
        train_samples.apply(lambda x: df_set(x, s))
        meaning_set = set(meaning_list)
        if s == set(meaning_set):
            print('all items included')

        else:
            print(meaning_set.difference(s))
            print('add all items manually')
            # # solution 1: more overlap
            # flgs = [0]*len(meaning_list)
            # selected_idx = []
            # selected_samples = []
            # for i in range(len(meaning_list)):
            #     if flgs[i] == 0:
            #         idx, samp = find_idx_from_df(meaning_list[i], meaning_samples)
            #         selected_idx.append(idx)
            #         selected_samples.append(samp)
            #         update_flg_for_list(samp, meaning_list, flgs)


            #Solution 2: less overlap
            verb = [i for i in meaning_list if 'VERB_' in i]
            random.shuffle(verb)
            verb_ = set(verb)
            noun = [i for i in meaning_list if 'NAME_' in i]
            random.shuffle(noun)
            noun_ = set(noun)

            sample_select = []
            for i in range(min(len(noun)//2, len(verb))):
                sample_select.append([noun[2*i], verb[i],noun[2*i+1]])
                noun_.discard(noun[2*i])
                noun_.discard(noun[2*i+1])
                verb_.discard(verb[i])

            if len(verb_) > 0:
                for v in verb_:
                    if len(noun_) > 0:
                        n1 = noun_.pop()
                        tmp_noun = noun.copy()
                        tmp_noun.pop(tmp_noun.index(n1))
                        n2 = tmp_noun[random.randint(0, len(tmp_noun))]
                        sample_select.append([n1, v, n2])
                    else:
                        tmp_noun = noun.copy()
                        n1 = tmp_noun[random.randint(0, len(tmp_noun))]
                        tmp_noun.pop(tmp_noun.index(n1))
                        n2 = tmp_noun[random.randint(0, len(tmp_noun))]
                        sample_select.append([n1, v, n2])
            else:
                for n in noun_:
                    noun_.discard(n)
                    v = verb[random.randint(0, len(verb))]
                    if len(noun_) > 0:
                        n2 = noun_.pop()
                    else:
                        tmp_noun = noun.copy()
                        tmp_noun.pop(tmp_noun.index(n))
                        n2 = tmp_noun[random.randint(0, len(tmp_noun))]
                    sample_select.append([n, v, n2])

            selected_idx = []
            for s in sample_select:
                idx, samp = find_idx_from_df_equal(s, meaning_samples)
                selected_idx.append(idx)

            for idx in selected_idx:
                indices.insert(0, indices.pop(indices.index(idx)))

    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def find_idx_from_df(item, samples):
    for i in range(len(samples)):
        if item in samples.iloc[i]:
            return i, samples.iloc[i]
    return -1, None

def find_idx_from_df_equal(item, samples):
    for i in range(len(samples)):
        if item == samples.iloc[i]:
            return i, samples.iloc[i]
    return -1, None

def v2_find_idx_from_df(item, samples, item_list, flgs):
    for i in range(len(samples)):
        if item in samples.iloc[i]:
            it1, it2, it3 = samples.iloc[i]
            return i, samples.iloc[i]
    return -1, None

def update_flg_for_list(items, samples, flgs):
    if len(samples) != len(flgs):
        print('ERROR FLG')
        return flgs

    for item in items:
        for i in range(len(samples)):
            if item == samples[i]:
                flgs[i] = 1
    return flgs
