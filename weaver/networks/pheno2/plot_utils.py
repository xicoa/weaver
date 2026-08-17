import matplotlib.pyplot as plt

import numpy as np

label_list = ["label_X_bb", "label_X_cc", "label_X_ss", "label_X_qq", "label_X_bc", "label_X_cs", "label_X_bq", "label_X_cq", "label_X_sq", "label_X_gg", "label_X_ee", "label_X_mm", "label_X_tauhtaue", "label_X_tauhtaum", "label_X_tauhtauh", "label_X_YY_bbbb", "label_X_YY_bbcc", "label_X_YY_bbss", "label_X_YY_bbqq", "label_X_YY_bbgg", "label_X_YY_bbee", "label_X_YY_bbmm", "label_X_YY_bbtauhtaue", "label_X_YY_bbtauhtaum", "label_X_YY_bbtauhtauh", "label_X_YY_bbb", "label_X_YY_bbc", "label_X_YY_bbs", "label_X_YY_bbq", "label_X_YY_bbg", "label_X_YY_bbe", "label_X_YY_bbm", "label_X_YY_cccc", "label_X_YY_ccss", "label_X_YY_ccqq", "label_X_YY_ccgg", "label_X_YY_ccee", "label_X_YY_ccmm", "label_X_YY_cctauhtaue", "label_X_YY_cctauhtaum", "label_X_YY_cctauhtauh", "label_X_YY_ccb", "label_X_YY_ccc", "label_X_YY_ccs", "label_X_YY_ccq", "label_X_YY_ccg", "label_X_YY_cce", "label_X_YY_ccm", "label_X_YY_ssss", "label_X_YY_ssqq", "label_X_YY_ssgg", "label_X_YY_ssee", "label_X_YY_ssmm", "label_X_YY_sstauhtaue", "label_X_YY_sstauhtaum", "label_X_YY_sstauhtauh", "label_X_YY_ssb", "label_X_YY_ssc", "label_X_YY_sss", "label_X_YY_ssq", "label_X_YY_ssg", "label_X_YY_sse", "label_X_YY_ssm", "label_X_YY_qqqq", "label_X_YY_qqgg", "label_X_YY_qqee", "label_X_YY_qqmm", "label_X_YY_qqtauhtaue", "label_X_YY_qqtauhtaum", "label_X_YY_qqtauhtauh", "label_X_YY_qqb", "label_X_YY_qqc", "label_X_YY_qqs", "label_X_YY_qqq", "label_X_YY_qqg", "label_X_YY_qqe", "label_X_YY_qqm", "label_X_YY_gggg", "label_X_YY_ggee", "label_X_YY_ggmm", "label_X_YY_ggtauhtaue", "label_X_YY_ggtauhtaum", "label_X_YY_ggtauhtauh", "label_X_YY_ggb", "label_X_YY_ggc", "label_X_YY_ggs", "label_X_YY_ggq", "label_X_YY_ggg", "label_X_YY_gge", "label_X_YY_ggm", "label_X_YY_bee", "label_X_YY_cee", "label_X_YY_see", "label_X_YY_qee", "label_X_YY_gee", "label_X_YY_bmm", "label_X_YY_cmm", "label_X_YY_smm", "label_X_YY_qmm", "label_X_YY_gmm", "label_X_YY_btauhtaue", "label_X_YY_ctauhtaue", "label_X_YY_stauhtaue", "label_X_YY_qtauhtaue", "label_X_YY_gtauhtaue", "label_X_YY_btauhtaum", "label_X_YY_ctauhtaum", "label_X_YY_stauhtaum", "label_X_YY_qtauhtaum", "label_X_YY_gtauhtaum", "label_X_YY_btauhtauh", "label_X_YY_ctauhtauh", "label_X_YY_stauhtauh", "label_X_YY_qtauhtauh", "label_X_YY_gtauhtauh", "label_X_YY_qqqb", "label_X_YY_qqqc", "label_X_YY_qqqs", "label_X_YY_bbcq", "label_X_YY_ccbs", "label_X_YY_ccbq", "label_X_YY_ccsq", "label_X_YY_sscq", "label_X_YY_qqbc", "label_X_YY_qqbs", "label_X_YY_qqcs", "label_X_YY_bcsq", "label_X_YY_bcs", "label_X_YY_bcq", "label_X_YY_bsq", "label_X_YY_csq", "label_X_YY_bcev", "label_X_YY_csev", "label_X_YY_bqev", "label_X_YY_cqev", "label_X_YY_sqev", "label_X_YY_qqev", "label_X_YY_bcmv", "label_X_YY_csmv", "label_X_YY_bqmv", "label_X_YY_cqmv", "label_X_YY_sqmv", "label_X_YY_qqmv", "label_X_YY_bctauev", "label_X_YY_cstauev", "label_X_YY_bqtauev", "label_X_YY_cqtauev", "label_X_YY_sqtauev", "label_X_YY_qqtauev", "label_X_YY_bctaumv", "label_X_YY_cstaumv", "label_X_YY_bqtaumv", "label_X_YY_cqtaumv", "label_X_YY_sqtaumv", "label_X_YY_qqtaumv", "label_X_YY_bctauhv", "label_X_YY_cstauhv", "label_X_YY_bqtauhv", "label_X_YY_cqtauhv", "label_X_YY_sqtauhv", "label_X_YY_qqtauhv", "label_QCD_bbccss", "label_QCD_bbccs", "label_QCD_bbcc", "label_QCD_bbcss", "label_QCD_bbcs", "label_QCD_bbc", "label_QCD_bbss", "label_QCD_bbs", "label_QCD_bb", "label_QCD_bccss", "label_QCD_bccs", "label_QCD_bcc", "label_QCD_bcss", "label_QCD_bcs", "label_QCD_bc", "label_QCD_bss", "label_QCD_bs", "label_QCD_b", "label_QCD_ccss", "label_QCD_ccs", "label_QCD_cc", "label_QCD_css", "label_QCD_cs", "label_QCD_c", "label_QCD_ss", "label_QCD_s", "label_QCD_light"]

def visualize_aux_particles(ax, array, mask):
    radius_fn = lambda pt: np.sqrt(pt) / 200
    # auxiliary truth particles if this is a genjet
    for i in range(1, int(sum(mask[0]))):
        pt = np.exp((array[0][i] / 0.7) + 1.7)
        deta = array[-2][i]
        dphi = array[-1][i]
        if array[6][i] > 0: # ResX
            color = 'yellow'
        elif array[7][i] > 0: # ResY
            color = 'gold'
        elif array[8][i] > 0: # ResDecayProd
            color = 'blue'
        elif array[9][i] > 0: # TauDecayProd
            color = 'violet'
        elif array[10][i] > 0: # TauDecayProd
            color = 'grey'
        else:
            raise ValueError('Unknown particle type')
        if color is None:
            continue
        pidlabel = 'U' if array[11][i] > 0 else \
            'D' if array[12][i] > 0 else \
            'S' if array[13][i] > 0 else \
            'C' if array[14][i] > 0 else \
            'B' if array[15][i] > 0 else \
            'G' if array[16][i] > 0 else \
            'El' if array[17][i] > 0 else \
            'Mu' if array[18][i] > 0 else \
            'Ta' if array[19][i] > 0 else \
            'Nu' if array[20][i] > 0 else \
            'Had' if array[21][i] > 0 else ''

        # draw squares for auxiliary particles
        ax.add_patch(plt.Circle((deta, dphi), radius_fn(pt), clip_on=True, alpha=0.2, facecolor=color))
        ax.text(deta, dphi, pidlabel, ha='center', va='center', fontsize=10) # annotate particle PDGIDs

    ax.set_xlabel(r'$\Delta\eta$'); ax.set_ylabel(r'$\Delta\phi$')
    xlim = 1.2
    ax.add_patch(plt.Circle((0, 0), 0.8, clip_on=True, facecolor='none', edgecolor='black', linestyle='--'))
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
    ax.set(xlabel=None); ax.set(ylabel=None)
    ax.set_aspect('equal')

def get_img_array(array, mask):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 5), dpi=50)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    # plotting
    visualize_aux_particles(ax, array, mask)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4)).transpose(2, 0, 1)[:3]
