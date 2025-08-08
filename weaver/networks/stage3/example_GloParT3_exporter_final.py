import os
import ast
import math
import torch
import torch.nn as nn
from torch import Tensor
import tqdm
import time
from collections import defaultdict, Counter
import numpy as np

from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformerTagger_ncoll

# Adapted from model in example_ParticleTransformer2024PlusTagger_unified2.py

class GlobalParticleTransformerExporter(nn.Module):
    def __init__(self, finetune_kw=dict(), **kwargs) -> None:
        '''
            finetune_kw (dict): fine-tuning configurations
            - mode (str): fine-tuning mode, 'cls' for classification, 'reg.guass' for regression with Gaussian NLL loss
            - input_highlevel_dim (int): dimension of the high-level input features
            - target_inds: list of target indices for the fine-tuning; can be a list of integers, a single integer, 'all', None
            - num_ft_nodes (int): number of output nodes of the external FC layer
            - freeze_main_params (bool): whether to freeze the main model parameters
            - fc_params (list): list of tuples (dim, dropout) of the FC layers
            - fc_suff_kw (dict): suffix FC configurations
                 - append_after (str): 'output', 'hidden', 'fc.0'
                 - params (list): list of tuples (dim, dropout) of the FC layers
        '''

        super().__init__()
        # special configs for exporter
        self.glopart_version = kwargs.pop('version')
        self.glopart_selected_indices = kwargs.pop('selected_indices', None)
        self.num_output_nodes = kwargs.pop('num_output_nodes')
        self.num_cls_nodes = kwargs.pop('num_cls_nodes')
        assert kwargs.get('for_inference') == True, 'for_inference must be True in exporter'

        # main model
        self.main = ParticleTransformerTagger_ncoll(**kwargs)

        # external FC
        self.mode = finetune_kw.get('mode') # mode of fine-tuning, determine which loss function etc to use
        self.input_highlevel_dim = finetune_kw.get('input_highlevel_dim')
        assert self.input_highlevel_dim > 0, 'High-level input dimension must be specified in exporter'

        self.target_inds = finetune_kw.get('target_inds')
        if self.target_inds == 'all':
            self.target_inds = list(range(kwargs['num_classes']))
        elif isinstance(self.target_inds, int):
            self.target_inds = [self.target_inds]
        assert finetune_kw.get('target_inds_opt', None) is None, 'Target indices option not supported in exporter'

        self.num_ft_nodes = finetune_kw.get('num_ft_nodes')
        assert finetune_kw.get('freeze_main_params', True) == True, 'Freezing main model params not supported in exporter'

        fc_params = finetune_kw.get('fc_params')
        assert finetune_kw.get('fc_suff_kw', None) is None, 'Suffix FC not supported in exporter'

        fcs = []
        in_dim = kwargs['embed_dims'][-1] + self.input_highlevel_dim # concat high-level input dims to the embed layer
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, self.num_ft_nodes)) # dim -> num_ft_nodes
        self.fc = nn.Sequential(*fcs)

    def forward(self, *args):
        # process main model
        output, x = self.main(*args[:-1])
        xcat = torch.cat([x, args[-1].squeeze(2)], dim=1)

        # process FC
        with torch.autocast('cuda', enabled=self.main.use_amp):
            output_fc = self.fc(xcat)

        # FC output as the residual to the main model output (after slicing)
        output_aux = output[:, self.target_inds] + output_fc

        # for exporter: 
        # do both softmax for the main model aux FC
        # then concat the output
        output_cls, output_rest = output.split([self.num_cls_nodes, output.size(1) - self.num_cls_nodes], dim=1)
        output_cls = torch.softmax(output_cls, dim=1)
        output = torch.cat([output_cls, output_rest], dim=1)
        output_aux = torch.softmax(output_aux, dim=1)

        output = torch.cat([output, output_aux], dim=1)

        return self.postprocess(x, output)

    def postprocess(self, x, output):

        if self.glopart_version in ['beta4p1', 'beta4p1:selected_indices']:
            # 374+2+374 output values

            class_labels = ['label_H_bb', 'label_H_cc', 'label_H_ss', 'label_H_qq', 'label_Hp_bc', 'label_Hm_bc', 'label_H_bs', 'label_Hp_cs', 'label_Hm_cs', 'label_H_gg', 'label_H_aa', 'label_H_ee', 'label_H_mm', 'label_H_tauhtaue', 'label_H_tauhtaum', 'label_H_tauhtauh', 'label_Top_bWpcs', 'label_Top_bWpqq', 'label_Top_bWpc', 'label_Top_bWps', 'label_Top_bWpq', 'label_Top_bWpev', 'label_Top_bWpmv', 'label_Top_bWptauev', 'label_Top_bWptaumv', 'label_Top_bWptauhv', 'label_Top_Wpcs', 'label_Top_Wpqq', 'label_Top_Wpev', 'label_Top_Wpmv', 'label_Top_Wptauev', 'label_Top_Wptaumv', 'label_Top_Wptauhv', 'label_Top_bWmcs', 'label_Top_bWmqq', 'label_Top_bWmc', 'label_Top_bWms', 'label_Top_bWmq', 'label_Top_bWmev', 'label_Top_bWmmv', 'label_Top_bWmtauev', 'label_Top_bWmtaumv', 'label_Top_bWmtauhv', 'label_Top_Wmcs', 'label_Top_Wmqq', 'label_Top_Wmev', 'label_Top_Wmmv', 'label_Top_Wmtauev', 'label_Top_Wmtaumv', 'label_Top_Wmtauhv', 'label_H_WW_cscs', 'label_H_WW_csqq', 'label_H_WW_qqqq', 'label_H_WW_csc', 'label_H_WW_css', 'label_H_WW_csq', 'label_H_WW_qqc', 'label_H_WW_qqs', 'label_H_WW_qqq', 'label_H_WW_csev', 'label_H_WW_qqev', 'label_H_WW_csmv', 'label_H_WW_qqmv', 'label_H_WW_cstauev', 'label_H_WW_qqtauev', 'label_H_WW_cstaumv', 'label_H_WW_qqtaumv', 'label_H_WW_cstauhv', 'label_H_WW_qqtauhv', 'label_H_WxWx_cscs', 'label_H_WxWx_csqq', 'label_H_WxWx_qqqq', 'label_H_WxWx_csc', 'label_H_WxWx_css', 'label_H_WxWx_csq', 'label_H_WxWx_qqc', 'label_H_WxWx_qqs', 'label_H_WxWx_qqq', 'label_H_WxWx_csev', 'label_H_WxWx_qqev', 'label_H_WxWx_csmv', 'label_H_WxWx_qqmv', 'label_H_WxWx_cstauev', 'label_H_WxWx_qqtauev', 'label_H_WxWx_cstaumv', 'label_H_WxWx_qqtaumv', 'label_H_WxWx_cstauhv', 'label_H_WxWx_qqtauhv', 'label_H_WxWxStar_cscs', 'label_H_WxWxStar_csqq', 'label_H_WxWxStar_qqqq', 'label_H_WxWxStar_csc', 'label_H_WxWxStar_css', 'label_H_WxWxStar_csq', 'label_H_WxWxStar_qqc', 'label_H_WxWxStar_qqs', 'label_H_WxWxStar_qqq', 'label_H_WxWxStar_csev', 'label_H_WxWxStar_qqev', 'label_H_WxWxStar_csmv', 'label_H_WxWxStar_qqmv', 'label_H_WxWxStar_cstauev', 'label_H_WxWxStar_qqtauev', 'label_H_WxWxStar_cstaumv', 'label_H_WxWxStar_qqtaumv', 'label_H_WxWxStar_cstauhv', 'label_H_WxWxStar_qqtauhv', 'label_H_ZZ_bbbb', 'label_H_ZZ_bbcc', 'label_H_ZZ_bbss', 'label_H_ZZ_bbqq', 'label_H_ZZ_cccc', 'label_H_ZZ_ccss', 'label_H_ZZ_ccqq', 'label_H_ZZ_ssss', 'label_H_ZZ_ssqq', 'label_H_ZZ_qqqq', 'label_H_ZZ_bbb', 'label_H_ZZ_bbc', 'label_H_ZZ_bbs', 'label_H_ZZ_bbq', 'label_H_ZZ_ccb', 'label_H_ZZ_ccc', 'label_H_ZZ_ccs', 'label_H_ZZ_ccq', 'label_H_ZZ_ssb', 'label_H_ZZ_ssc', 'label_H_ZZ_sss', 'label_H_ZZ_ssq', 'label_H_ZZ_qqb', 'label_H_ZZ_qqc', 'label_H_ZZ_qqs', 'label_H_ZZ_qqq', 'label_H_ZZ_bbee', 'label_H_ZZ_bbmm', 'label_H_ZZ_bbe', 'label_H_ZZ_bbm', 'label_H_ZZ_bee', 'label_H_ZZ_bmm', 'label_H_ZZ_bbtauhtaue', 'label_H_ZZ_bbtauhtaum', 'label_H_ZZ_bbtauhtauh', 'label_H_ZZ_btauhtaue', 'label_H_ZZ_btauhtaum', 'label_H_ZZ_btauhtauh', 'label_H_ZZ_ccee', 'label_H_ZZ_ccmm', 'label_H_ZZ_cce', 'label_H_ZZ_ccm', 'label_H_ZZ_cee', 'label_H_ZZ_cmm', 'label_H_ZZ_cctauhtaue', 'label_H_ZZ_cctauhtaum', 'label_H_ZZ_cctauhtauh', 'label_H_ZZ_ctauhtaue', 'label_H_ZZ_ctauhtaum', 'label_H_ZZ_ctauhtauh', 'label_H_ZZ_ssee', 'label_H_ZZ_ssmm', 'label_H_ZZ_sse', 'label_H_ZZ_ssm', 'label_H_ZZ_see', 'label_H_ZZ_smm', 'label_H_ZZ_sstauhtaue', 'label_H_ZZ_sstauhtaum', 'label_H_ZZ_sstauhtauh', 'label_H_ZZ_stauhtaue', 'label_H_ZZ_stauhtaum', 'label_H_ZZ_stauhtauh', 'label_H_ZZ_qqee', 'label_H_ZZ_qqmm', 'label_H_ZZ_qqe', 'label_H_ZZ_qqm', 'label_H_ZZ_qee', 'label_H_ZZ_qmm', 'label_H_ZZ_qqtauhtaue', 'label_H_ZZ_qqtauhtaum', 'label_H_ZZ_qqtauhtauh', 'label_H_ZZ_qtauhtaue', 'label_H_ZZ_qtauhtaum', 'label_H_ZZ_qtauhtauh', 'label_H_ZxZx_bbbb', 'label_H_ZxZx_bbcc', 'label_H_ZxZx_bbss', 'label_H_ZxZx_bbqq', 'label_H_ZxZx_cccc', 'label_H_ZxZx_ccss', 'label_H_ZxZx_ccqq', 'label_H_ZxZx_ssss', 'label_H_ZxZx_ssqq', 'label_H_ZxZx_qqqq', 'label_H_ZxZx_bbb', 'label_H_ZxZx_bbc', 'label_H_ZxZx_bbs', 'label_H_ZxZx_bbq', 'label_H_ZxZx_ccb', 'label_H_ZxZx_ccc', 'label_H_ZxZx_ccs', 'label_H_ZxZx_ccq', 'label_H_ZxZx_ssb', 'label_H_ZxZx_ssc', 'label_H_ZxZx_sss', 'label_H_ZxZx_ssq', 'label_H_ZxZx_qqb', 'label_H_ZxZx_qqc', 'label_H_ZxZx_qqs', 'label_H_ZxZx_qqq', 'label_H_ZxZx_bbee', 'label_H_ZxZx_bbmm', 'label_H_ZxZx_bbe', 'label_H_ZxZx_bbm', 'label_H_ZxZx_bee', 'label_H_ZxZx_bmm', 'label_H_ZxZx_bbtauhtaue', 'label_H_ZxZx_bbtauhtaum', 'label_H_ZxZx_bbtauhtauh', 'label_H_ZxZx_btauhtaue', 'label_H_ZxZx_btauhtaum', 'label_H_ZxZx_btauhtauh', 'label_H_ZxZx_ccee', 'label_H_ZxZx_ccmm', 'label_H_ZxZx_cce', 'label_H_ZxZx_ccm', 'label_H_ZxZx_cee', 'label_H_ZxZx_cmm', 'label_H_ZxZx_cctauhtaue', 'label_H_ZxZx_cctauhtaum', 'label_H_ZxZx_cctauhtauh', 'label_H_ZxZx_ctauhtaue', 'label_H_ZxZx_ctauhtaum', 'label_H_ZxZx_ctauhtauh', 'label_H_ZxZx_ssee', 'label_H_ZxZx_ssmm', 'label_H_ZxZx_sse', 'label_H_ZxZx_ssm', 'label_H_ZxZx_see', 'label_H_ZxZx_smm', 'label_H_ZxZx_sstauhtaue', 'label_H_ZxZx_sstauhtaum', 'label_H_ZxZx_sstauhtauh', 'label_H_ZxZx_stauhtaue', 'label_H_ZxZx_stauhtaum', 'label_H_ZxZx_stauhtauh', 'label_H_ZxZx_qqee', 'label_H_ZxZx_qqmm', 'label_H_ZxZx_qqe', 'label_H_ZxZx_qqm', 'label_H_ZxZx_qee', 'label_H_ZxZx_qmm', 'label_H_ZxZx_qqtauhtaue', 'label_H_ZxZx_qqtauhtaum', 'label_H_ZxZx_qqtauhtauh', 'label_H_ZxZx_qtauhtaue', 'label_H_ZxZx_qtauhtaum', 'label_H_ZxZx_qtauhtauh', 'label_H_ZxZxStar_bbbb', 'label_H_ZxZxStar_bbcc', 'label_H_ZxZxStar_bbss', 'label_H_ZxZxStar_bbqq', 'label_H_ZxZxStar_cccc', 'label_H_ZxZxStar_ccss', 'label_H_ZxZxStar_ccqq', 'label_H_ZxZxStar_ssss', 'label_H_ZxZxStar_ssqq', 'label_H_ZxZxStar_qqqq', 'label_H_ZxZxStar_bbb', 'label_H_ZxZxStar_bbc', 'label_H_ZxZxStar_bbs', 'label_H_ZxZxStar_bbq', 'label_H_ZxZxStar_ccb', 'label_H_ZxZxStar_ccc', 'label_H_ZxZxStar_ccs', 'label_H_ZxZxStar_ccq', 'label_H_ZxZxStar_ssb', 'label_H_ZxZxStar_ssc', 'label_H_ZxZxStar_sss', 'label_H_ZxZxStar_ssq', 'label_H_ZxZxStar_qqb', 'label_H_ZxZxStar_qqc', 'label_H_ZxZxStar_qqs', 'label_H_ZxZxStar_qqq', 'label_H_ZxZxStar_bbee', 'label_H_ZxZxStar_bbmm', 'label_H_ZxZxStar_bbe', 'label_H_ZxZxStar_bbm', 'label_H_ZxZxStar_bee', 'label_H_ZxZxStar_bmm', 'label_H_ZxZxStar_bbtauhtaue', 'label_H_ZxZxStar_bbtauhtaum', 'label_H_ZxZxStar_bbtauhtauh', 'label_H_ZxZxStar_btauhtaue', 'label_H_ZxZxStar_btauhtaum', 'label_H_ZxZxStar_btauhtauh', 'label_H_ZxZxStar_ccee', 'label_H_ZxZxStar_ccmm', 'label_H_ZxZxStar_cce', 'label_H_ZxZxStar_ccm', 'label_H_ZxZxStar_cee', 'label_H_ZxZxStar_cmm', 'label_H_ZxZxStar_cctauhtaue', 'label_H_ZxZxStar_cctauhtaum', 'label_H_ZxZxStar_cctauhtauh', 'label_H_ZxZxStar_ctauhtaue', 'label_H_ZxZxStar_ctauhtaum', 'label_H_ZxZxStar_ctauhtauh', 'label_H_ZxZxStar_ssee', 'label_H_ZxZxStar_ssmm', 'label_H_ZxZxStar_sse', 'label_H_ZxZxStar_ssm', 'label_H_ZxZxStar_see', 'label_H_ZxZxStar_smm', 'label_H_ZxZxStar_sstauhtaue', 'label_H_ZxZxStar_sstauhtaum', 'label_H_ZxZxStar_sstauhtauh', 'label_H_ZxZxStar_stauhtaue', 'label_H_ZxZxStar_stauhtaum', 'label_H_ZxZxStar_stauhtauh', 'label_H_ZxZxStar_qqee', 'label_H_ZxZxStar_qqmm', 'label_H_ZxZxStar_qqe', 'label_H_ZxZxStar_qqm', 'label_H_ZxZxStar_qee', 'label_H_ZxZxStar_qmm', 'label_H_ZxZxStar_qqtauhtaue', 'label_H_ZxZxStar_qqtauhtaum', 'label_H_ZxZxStar_qqtauhtauh', 'label_H_ZxZxStar_qtauhtaue', 'label_H_ZxZxStar_qtauhtaum', 'label_H_ZxZxStar_qtauhtauh', 'label_H_HV_aabb', 'label_H_HV_aacc', 'label_H_HV_aass', 'label_H_HV_aaqq', 'label_H_HV_aabc', 'label_H_HV_aacs', 'label_H_HV_aabq', 'label_H_HV_aacq', 'label_H_HV_aasq', 'label_H_HV_aagg', 'label_H_HV_aaee', 'label_H_HV_aamm', 'label_H_HV_aatauhtaue', 'label_H_HV_aatauhtaum', 'label_H_HV_aatauhtauh', 'label_H_HV_aab', 'label_H_HV_aac', 'label_H_HV_aas', 'label_H_HV_aaq', 'label_H_HV_aag', 'label_H_HV_aae', 'label_H_HV_aam', 'label_H_HV_aataue', 'label_H_HV_aataum', 'label_H_HV_aatauh', 'label_H_HV_abb', 'label_H_HV_acc', 'label_H_HV_ass', 'label_H_HV_aqq', 'label_H_HV_abc', 'label_H_HV_acs', 'label_H_HV_abq', 'label_H_HV_acq', 'label_H_HV_asq', 'label_H_HV_agg', 'label_H_HV_aee', 'label_H_HV_amm', 'label_H_HV_atauhtaue', 'label_H_HV_atauhtaum', 'label_H_HV_atauhtauh', 'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others']
            aux_class_labels = ['label_Top_bWcs', 'label_Top_bWqq', 'label_Top_bWc', 'label_Top_bWs', 'label_Top_bWq', 'label_Top_bWev', 'label_Top_bWmv', 'label_Top_bWtauev', 'label_Top_bWtaumv', 'label_Top_bWtauhv', 'label_Top_Wcs', 'label_Top_Wqq', 'label_Top_Wev', 'label_Top_Wmv', 'label_Top_Wtauev', 'label_Top_Wtaumv', 'label_Top_Wtauhv', 'label_W_cs', 'label_W_qq', 'label_Z_bb', 'label_Z_cc', 'label_Z_ss', 'label_Z_qq', 'label_QCD_bb', 'label_QCD_cc', 'label_QCD_b', 'label_QCD_c', 'label_QCD_others']

            # assign output variable name (374+2+374+28 outputs)
            output_varname = [l.replace('_', '').replace('label', 'prob') for l in class_labels]
            output_varname += ['massCorrResonance', 'massCorrGeneric']
            output_varname += [l.replace('_', '').replace('label', 'massCorr') for l in class_labels]
            output_varname += [l.replace('_', '').replace('label', 'probWithMass') for l in aux_class_labels]

            assert len(output_varname) == 374+2+374+28 # beta4 setup
            
            if self.glopart_version == 'beta4p1:selected_indices':
                # only select nessesary nodes
                assert self.glopart_selected_indices is not None, 'Selected indices must be provided for selected_fc'
                output_varname = [output_varname[:750][i] for i in self.glopart_selected_indices] + output_varname[750:]

            # define variable replacement map
            replace_map = {name: f'output[:,{i}]' for i, name in enumerate(output_varname)}

            def get_expr(expression):
                tree = ast.parse(expression, mode='eval')
                # variables = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                for n in ast.walk(tree):
                    if isinstance(n, ast.Name) and n.id in replace_map:
                        n.id = replace_map[n.id]
                return ast.unparse(tree)

            output_nodes_expr = [
                ## ===== classification nodes ===== #
                'probHbb', # Xbb
                'probHcc', # Xcc
                'probHpcs + probHmcs', # Xcs
                'probHss + probHqq*2', # Xqq
                'probHtauhtaue', # Xtauhtaue
                'probHtauhtaum', # Xtauhtaum
                'probHtauhtauh', # Xtauhtauh
                'probHWWcscs + probHWWcsqq + probHWWqqqq', # XWW4q
                'probHWWcsc + probHWWcss + probHWWcsq + probHWWqqc + probHWWqqs + probHWWqqq', # XWW3q
                'probHWWcsev + probHWWqqev', # XWWqqev
                'probHWWcsmv + probHWWqqmv', # XWWqqmv
                'probTopbWpcs + probTopbWpqq + probTopbWmcs + probTopbWmqq', # TopbWqq
                'probTopbWpc + probTopbWps + probTopbWpq + probTopbWmc + probTopbWms + probTopbWmq', # TopbWq
                'probTopbWpev + probTopbWmev', # TopbWev
                'probTopbWpmv + probTopbWmmv', # TopbWmv
                'probTopbWptauhv + probTopbWmtauhv', # TopbWtauhv
                'probQCDbb + probQCDcc + probQCDb + probQCDc + probQCDothers', # QCD
                ## ===== regression nodes ===== ##
                'massCorrGeneric + (massCorrHbb*probHbb + massCorrHcc*probHcc + massCorrHss*probHss + 2*massCorrHqq*probHqq + massCorrHpcs*probHpcs + massCorrHmcs*probHmcs) / (probHbb + probHcc + probHss + 2*probHqq + probHpcs + probHmcs).clamp(min=1e-10)', # massCorrX2p
                'massCorrGeneric', # massCorrGeneric
                ## ===== regression nodes for non-MD tagger ===== ##
                '(probWithMassTopbWcs + probWithMassTopbWqq) / (probWithMassTopbWcs + probWithMassTopbWqq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers).clamp(min=1e-10)', # probWithMassTopvsQCD
                '(probWithMassWcs + probWithMassWqq) / (probWithMassWcs + probWithMassWqq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers).clamp(min=1e-10)', # probWithMassWvsQCD
                '(probWithMassZbb + probWithMassZcc + probWithMassZss + probWithMassZqq) / (probWithMassZbb + probWithMassZcc + probWithMassZss + probWithMassZqq + probWithMassQCDbb + probWithMassQCDcc + probWithMassQCDb + probWithMassQCDc + probWithMassQCDothers).clamp(min=1e-10)', # probWithMassZvsQCD
            ]
            assert len(output_nodes_expr) == self.num_output_nodes, f"Output nodes mismatch: {len(output_nodes_expr)} != {self.num_output_nodes}"
            
            # calculate output nodes
            output_nodes = []
            for i, expr in enumerate(output_nodes_expr):
                output_nodes.append(eval(get_expr(expr)))
            output_nodes = torch.stack(output_nodes, dim=1)
            return torch.concat([output_nodes, x], dim=1) # append hidden layer x after output nodes

        else:
            raise NotImplementedError(f"Unsupported version: {self.glopart_version}")

# Adapted from get_model in example_ParticleTransformer2024PlusTagger_unified2.py
def get_model(data_config, **kwargs):
    assert 'num_nodes' in kwargs, 'num_nodes must be provided'
    assert 'num_cls_nodes' in kwargs, 'num_cls_nodes must be provided'
    num_nodes = kwargs.pop('num_nodes')
    num_cls_nodes = kwargs.pop('num_cls_nodes')
    label_cls_nodes = kwargs.pop('label_cls_nodes', None)
    reg_kw = kwargs.pop('reg_kw', dict())
    finetune_kw = kwargs.pop('finetune_kw', None)
    eval_kw = kwargs.pop('eval_kw', dict())

    # use SwiGLU-default setup
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=num_nodes,
        # network configurations
        pair_input_type='pp',
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pair_norm=False,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={},
        fc_params=(),
        activation='gelu',
        # GloParT wrapper configurations
        # input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
        # use_external_fc=False,
        # misc
        trim=True,
        for_inference=False,
    )

    if kwargs.pop('use_swiglu_config', False):
        cfg.update(
            block_params={"scale_attn_mask": True, "scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
            cls_block_params={"scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        )
    if kwargs.pop('use_pair_norm_config', False):
        cfg.update(
            use_pair_norm=True,
            pair_input_dim=6,
        )

    cfg.update(**kwargs)

    # for exporter: finetune mode should be on
    assert finetune_kw.get('mode') is not None, 'mode must be provided in finetune_kw'
    finetune_kw.update(
        input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
    )
    cfg.update(
        finetune_kw=finetune_kw,
        return_embed=True, # return the last embed layer before FC
    )
    # special for exporter - to interface with ParticleTransformer2024Plus
    # do not apply softmax here!
    cfg.update(
        export_params={'apply_softmax': False, 'concat_hid': False},
        num_cls_nodes=num_cls_nodes, # sent this additional param to the exporter
    )
    model = GlobalParticleTransformerExporter(**cfg)

    # # set special args
    # model.num_nodes = num_nodes
    # model.num_cls_nodes = num_cls_nodes
    # model.eval_kw = eval_kw

    _logger.info('Model config: %s' % str(cfg))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }

    return model, model_info
