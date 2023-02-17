# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import json
import os
import pandas as pd
import torch

from egg.core import Callback, ConsoleLogger, Interaction
from egg.core import EarlyStopperAccuracy
from egg.core.early_stopping import EarlyStopper


def get_callbacks(log_dir, acc_threshhold, dump_output, outputs_dir, dump_every, save_model_dir, is_distributed: bool = False) -> List[Callback]:
    callbacks = [
        ConsoleLogSaver(as_json=True, print_train_loss=True, save2file=True, log_dir=log_dir, dump_output=dump_output, outputs_dir=outputs_dir, dump_every=dump_every),
        BestStatsTracker(),
        EarlyStopperAccuracy(acc_threshhold),
        GeneralModelSaver(save_model_dir),
    ]

    return callbacks


def v2_get_callbacks(log_dir, acc_threshhold, patience, dump_output, outputs_dir, dump_every, save_model_dir, is_distributed: bool = False) -> List[Callback]:
    callbacks = [
        ConsoleLogSaver(as_json=True, print_train_loss=True, save2file=True, log_dir=log_dir, dump_output=dump_output, outputs_dir=outputs_dir, dump_every=dump_every),
        BestStatsTracker(),
        EarlyStopping_NoImprovement(threshold=acc_threshhold, patience=patience),
        GeneralModelSaver(save_model_dir),
    ]
    return callbacks


def v3_get_callbacks_no_earlystop(log_dir, acc_threshhold, patience, dump_output, outputs_dir, dump_every, save_model_dir, is_distributed: bool = False) -> List[Callback]:
    print('no early stopping')
    callbacks = [
        ConsoleLogSaver(as_json=True, print_train_loss=True, save2file=True, log_dir=log_dir, dump_output=dump_output, outputs_dir=outputs_dir, dump_every=dump_every),
        BestStatsTracker(),
        # EarlyStopping_NoImprovement(threshold=acc_threshhold, patience=patience),
        GeneralModelSaver(save_model_dir),
    ]
    return callbacks


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()

        # TRAIN
        self.best_train_acc, self.best_train_loss, self.best_train_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_train_acc, self.last_train_loss, self.last_train_epoch = 0.0, 0.0, 0
        # last_val_epoch useful for runs that end before the final epoch

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = _loss

        self.last_train_acc = logs.aux["acc"].mean().item()
        self.last_train_epoch = epoch
        self.last_train_loss = _loss

    def on_train_end(self):
        is_distributed = self.trainer.distributed_context.is_distributed
        is_leader = self.trainer.distributed_context.is_leader
        if (not is_distributed) or (is_distributed and is_leader):
            train_stats = dict(
                mode="train",
                epoch=self.best_train_epoch,
                acc=self.best_train_acc,
                loss=self.best_train_loss,
            )
            print(json.dumps(train_stats), flush=True)


class ConsoleLogSaver(ConsoleLogger):
    def __init__(self, print_train_loss=False, as_json=False, save2file=True, log_dir=None, dump_output=True, outputs_dir=None, dump_every=50):
        super(ConsoleLogSaver, self).__init__(print_train_loss, as_json)
        if save2file:
            self.log_dir = log_dir
        if dump_output:
            self.outputs_dir = outputs_dir
            self.dump_every = dump_every
        self.validation_outputs = []
        self.aggregate_log = []


    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        self.aggregate_log.append(dump)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            if logs.aux_input and 'order' in logs.aux_input.keys():
                order = logs.aux_input['order'].sum().item()
                dump.update(dict(order=order))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_train_end(self):
        df = pd.DataFrame.from_dict(self.aggregate_log)
        df.to_csv(self.log_dir, sep='\t')
        print(f'Log file saved to {self.log_dir}')

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)
        if epoch % self.dump_every == 0 or epoch < 10:
            dump_dir_mean = os.path.join(self.outputs_dir, f'mean_epoch{epoch}')
            dump_dir_uttr = os.path.join(self.outputs_dir, f'uttr_epoch{epoch}')
            dump_dir_msg = os.path.join(self.outputs_dir, f'msg_epoch{epoch}')
            dump_dir_lstpredict = os.path.join(self.outputs_dir, f'lstpred_epoch{epoch}')
            torch.save(logs.sender_input, dump_dir_mean)
            torch.save(logs.labels, dump_dir_uttr)
            torch.save(logs.message, dump_dir_msg)
            torch.save(logs.receiver_output, dump_dir_lstpredict)
        else:
            pass

    def on_early_stopping(
            self,
            train_loss: float,
            train_logs: Interaction,
            epoch: int,
            test_loss: float = None,
            test_logs: Interaction = None,
    ):
        print(f'early stopping at epoch {epoch}')
        # dump_dir_msg = os.path.join(self.outputs_dir, f'earlystop_msg_epoch{epoch}')
        # dump_dir_input = os.path.join(self.outputs_dir, f'earlystop_input_epoch{epoch}')
        # dump_dir_label = os.path.join(self.outputs_dir, f'earlystop_label_epoch{epoch}')
        # torch.save(test_logs.message, dump_dir_msg)
        # torch.save(test_logs.sender_input, dump_dir_input)
        # torch.save(test_logs.labels, dump_dir_label)

        dump_dir_mean = os.path.join(self.outputs_dir, f'earlystop_mean_epoch{epoch}')
        dump_dir_uttr = os.path.join(self.outputs_dir, f'earlystop_uttr_epoch{epoch}')
        dump_dir_msg = os.path.join(self.outputs_dir, f'earlystop_msg_epoch{epoch}')
        dump_dir_lstpredict = os.path.join(self.outputs_dir, f'earlystop_lstpred_epoch{epoch}')

        torch.save(test_logs.sender_input, dump_dir_mean)
        torch.save(test_logs.labels, dump_dir_uttr)
        torch.save(test_logs.message, dump_dir_msg)
        torch.save(test_logs.receiver_output, dump_dir_lstpredict)


class GeneralModelSaver(Callback):
    """A callback that stores module(s) in trainer's checkpoint_dir, if any."""

    def __init__(self, save_model_dir=None):
        super(GeneralModelSaver, self).__init__()
        self.save_model_dir = save_model_dir

    def save_model(self, epoch=""):
        model = self.trainer.game.model
        model_name = f"{model.name}_{epoch if epoch else 'final'}.pt"
        torch.save(model.state_dict(), os.path.join(self.save_model_dir, model_name))

    def on_train_end(self):
        self.save_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        self.save_model(epoch=epoch)


class EarlyStopping_NoImprovement(EarlyStopper):
    def __init__(self,
                 threshold: float, field_name: str = "acc", validation: bool = True,
                 min_delta=1e-5,
                 patience=5):
        super(EarlyStopping_NoImprovement, self).__init__(validation)
        self.threshold = threshold
        self.field_name = field_name
        # if "multi_acc": early_stopping use multi_acc as criterion, differ only for speaker
        # if "acc": early_stopping use normal acc

        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_acc = 0
        self.previous_acc = 0
        # self.stopped_epoch = 0

    def should_stop(self) -> bool:
        if self.validation:
            assert (
                self.validation_stats
            ), "Validation data must be provided for early stooping to work"
            loss, last_epoch_interactions = self.validation_stats[-1]
        else:
            assert (
                self.train_stats
            ), "Training data must be provided for early stooping to work"
            loss, last_epoch_interactions = self.train_stats[-1]

        metric_mean = last_epoch_interactions.aux[self.field_name].mean()
        acc_stop = metric_mean >= self.threshold

        noimprove_stop = False

        current_acc = metric_mean
        # metric_mean: normal acc

        if (current_acc - self.best_acc) > self.min_delta:
            self.best_acc = current_acc
            self.wait = 0  # reset
            noimprove_stop = False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # self.stopped_epoch = epoch + 1
                noimprove_stop = True

        # if (current_acc - self.previous_acc) > self.min_delta:
        #     self.wait = 0  # reset
        #     noimprove_stop = False
        # else:
        #     self.wait += 1
        #     if self.wait >= self.patience:
        #         # self.stopped_epoch = epoch + 1
        #         noimprove_stop = True
        # self.previous_acc = current_acc

        stop = acc_stop or noimprove_stop

        if stop:
            print(f'validation: {self.validation}')
            print(f'no_improve_stop:{noimprove_stop}')
            print(f'acc_stop:{acc_stop}')

        return stop

