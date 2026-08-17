import numpy as np
import awkward as ak
import torch
import tqdm
import time
import os
from collections import defaultdict, Counter

import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

# from weaver.nn.model.ParticleTransformer import ParticleTransformer
from utils.import_tools import import_module
import os
ParticleTransformer = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformer

from utils.logger import _logger
from utils.nn.tools import (
    _concat,
    AllGather,
    train_classification,
)

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


def get_train_fn(data_config, **kwargs):
    return train_classification


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_jetclass


def evaluate_classification_jetclass(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, extra_args=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                logits = AllGather.apply(model(*inputs))
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper:
        label_list = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
        scores_dict = {lab: scores[:, i] for i, lab in enumerate(label_list)}
        flag_dict = {lab: labels[data_config.label_names[0]] == i for i, lab in enumerate(label_list)}
        comp_list = [ # (sig, bkg, sig_eff)
            ('Hbb', 'QCD', 0.5),
            ('Hcc', 'QCD', 0.5),
            ('Hgg', 'QCD', 0.5),
            ('H4q', 'QCD', 0.5),
            ('Hqql','QCD',  0.99),
            ('Tbqq','QCD',  0.5),
            ('Tbl', 'QCD', 0.995),
            ('Wqq', 'QCD', 0.5),
            ('Zqq', 'QCD', 0.5),
        ]
        bkgrej, bkgrej_def = {}, {}
        def bkg_rej(tpr, fpr, eff=0.5): # default defination in ParT paper
            idx = next(idx for idx, v in enumerate(tpr) if v>eff)
            return 1/np.maximum(fpr[idx], 1e-8)

        f, ax = plt.subplots(figsize=(5, 5))

        for name_sig, name_bkg, sig_eff in comp_list:
            discr = scores_dict[name_sig] / (scores_dict[name_sig] + scores_dict[name_bkg])
            discr_sig, discr_bkg = discr[flag_dict[name_sig]], discr[flag_dict[name_bkg]]
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.ones_like(discr_sig), np.zeros_like(discr_bkg)]),
                np.concatenate([discr_sig, discr_bkg])
            )
            ax.plot(tpr, fpr, label='%s vs %s (AUC=%.4f)' % (name_sig, name_bkg, m.auc(fpr, tpr)))
            bkgrej[(name_sig, name_bkg)] = np.interp(sig_eff, tpr, 1. / np.maximum(fpr, 1e-8)) # bkgrej at sig_eff
            bkgrej_def[(name_sig, name_bkg)] = bkg_rej(tpr, fpr, sig_eff) # also calculate bkgrej with default defination

        ax.legend()
        ax.set_xlabel('True positive rate (signal eff.)', ha='right', x=1.0); ax.set_ylabel('False positive rate (BKG eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-5, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_helper.writer.add_figure('ROC/%s (epoch)' % tb_mode, f, epoch)

        # write bkgrej values
        for name_sig, name_bkg, sig_eff in comp_list:
            tb_helper.write_scalars([
                ('BkgRej/%s_vs_%s_sigeff%s/%s (epoch)' % (name_sig, name_bkg, sig_eff, tb_mode), bkgrej[(name_sig, name_bkg)] if bkgrej[(name_sig, name_bkg)] < 1e6 else np.nan, epoch),
                ('BkgRej_def/%s_vs_%s_sigeff%s/%s (epoch)' % (name_sig, name_bkg, sig_eff, tb_mode), bkgrej_def[(name_sig, name_bkg)] if bkgrej_def[(name_sig, name_bkg)] < 1e6 else np.nan, epoch),
            ])

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers
