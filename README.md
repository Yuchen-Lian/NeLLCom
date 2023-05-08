# NeLLCom: Neural-agent Language Learning and Communication

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

NeLLCom is a framework that allows researchers to quickly implement multi-agent miniature language learning games. 

In such games, the agents are firstly trained to listen or speak predefined languages via Supervised Learning (SL) 
and then pairs of speaking and listening agents talk to each other while optimizing communication success via Reinforcement Learning (RL).

The implementation of NeLLCom is partly based on EGG toolkit.

More details can be found in our TACL paper, 
titled "Communication Drives the Emergence of Language Universals in Neural Agents: Evidence from the Word-order/Case-marking Trade-off":
[arxiv](https://arxiv.org/abs/2301.13083)

## List of Explored Language Features

* Word-order/Case-marking Trade-off

## Agents Architecture

Speaking Agent
* Encoder: Linear
* Decoder: GRU

Listening Agent
* Encoder: GRU
* Decoder: Linear


## Installing NeLLCom

Generally, we assume that you use PyTorch 1.1.0 or newer and Python 3.6 or newer.

1. Installing [EGG](https://github.com/facebookresearch/EGG.git.) toolkit;
2. Moving to the EGG game design folder:
   ```
   cd EGG/egg/zoo
   ```
3. Cloning the NeLLCom into the EGG game design folder:
   ```
   git clone git@github.com:Yuchen-Lian/NeLLCom.git
   cd NeLLCom
   ```
4. Then, we can run a game, e.g. the Word-order/Case-marking trade-off game:
    ```bash
    python -m egg.zoo.NeLLCom.train --n_epochs=60
    ```

## NeLLCom structure

* `data/` contains the full dataset of the predefined artificial languages that are used in the paper.
* `train.py` contain the actual logic implementation.
* `games_*.py` contain the communication pipeline of the game.
* `archs_*.py` contain the agent stucture design.
* `pytorch-seq2seq/` is a git submodule containing a 3rd party seq2seq [framework](https://github.com/IBM/pytorch-seq2seq/).


## Citation
If you find NeLLCom useful in your research, please cite this paper:
```
@article{lian2023communication,
  title={Communication Drives the Emergence of Language Universals in Neural Agents: Evidence from the Word-order/Case-marking Trade-off},
  author={Lian, Yuchen and Bisazza, Arianna and Verhoef, Tessa},
  journal={arXiv preprint arXiv:2301.13083},
  year={2023}
}
```

## Licence
NeLLCom is licensed under MIT.
