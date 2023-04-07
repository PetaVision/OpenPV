import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from .AnalysisCommon import AnalysisCommon

from pvtools.readpvpfile import readpvpfile


def plot_lca_nnz(analysis_common: AnalysisCommon):
    # lca_filename = str(Path(analysis_common.checkpoint_path, 'S1Oracle_A.pvp'))
    #
    # lca_dict = readpvpfile(lca_filename)
    #
    # lca_data_tmp = lca_dict['values']
    # lca_header = lca_dict['header']

    print("Plotting LCA NNZ...")

    lca_header, lca_data_tmp = analysis_common.read_v1_a_file()

    num_lca = lca_data_tmp.shape[0]
    v_bins = analysis_common.v_thresh_bins[analysis_common.analysis_settings.lca_num_layers-1]
    lca_data = np.zeros((lca_header['nx'] * lca_header['ny'] * lca_header['nf'], num_lca))

    for i in range(num_lca):
        lca_data_slice = np.transpose(lca_data_tmp[i, :, :, :], (2, 1, 0))
        lca_data[:, i] = lca_data_slice.flatten(order='F')

    num_batch_skip = round(analysis_common.analysis_settings.num_batch / analysis_common.analysis_settings.skip_lca)
    num_lca_per_batch = math.floor(num_lca / num_batch_skip)
    lca_nnz = np.zeros((1, lca_header['nf']))
    lca_vals = np.zeros((1, len(analysis_common.v_thresh_bins[analysis_common.lca_index]) - 1))
    lca_size = (lca_header['nf'], lca_header['nx'], lca_header['ny'])
    max_lca_coef = -10000

    for i in range(num_lca):

        if np.count_nonzero(lca_data[:, i]) == 0:
            continue

        lca_indices = np.where(lca_data[:, i] != 0)
        lca_coef = lca_data[lca_indices, i]
        max_lca_coef = max(max_lca_coef, max(lca_coef[0, :]))
        lca_feature, lca_col, lca_row = np.unravel_index(lca_indices, lca_size, order='F')

        lca_nnz_new = np.histogram(lca_feature[0, :], bins=np.arange(0, lca_header['nf']+1, 1.0))[0]
        lca_nnz[0] += lca_nnz_new
        lc_hist = np.histogram(lca_coef[0, :], bins=v_bins)[0] / len(lca_coef[0, :])
        lca_vals[0, :] += lc_hist

    lca_nnz_normalized = lca_nnz / (lca_header['nx'] * lca_header['ny'] * num_lca)

    print("nnz(V{}) = {}".format(num_lca, np.mean(lca_nnz_normalized)))
    print("max_lca_coef(V{}) = {}".format(num_lca, max_lca_coef))

    lca_nnz_sorted = np.sort(lca_nnz_normalized[0])[::-1]
    lca_rank_index = lca_nnz_normalized[0].argsort()[::-1]

    title_str = "V{}_nnz_{}_{}_{}".format(num_lca,
                                                 analysis_common.analysis_settings.start_lca,
                                                 analysis_common.analysis_settings.skip_lca,
                                                 Path(analysis_common.checkpoint_path).name)

    fh_nnz, axs = plt.subplots(2, 1)
    fh_nnz.suptitle(title_str)

    lca_vals_normalized = lca_vals[0] / num_lca

    axs[1].bar(range(0, len(lca_vals_normalized)), lca_vals_normalized)
    axs[1].set_xticks(range(0, len(v_bins) + 1))
    xtick_labels = [''] * (len(v_bins) + 1)

    for i in range(0, len(v_bins) + 1, 4):
        xtick_labels[i] = "{:4.2f}".format(v_bins[i] + 0.125)

    axs[1].set_xticklabels(xtick_labels)

    axs[0].bar(range(0, len(lca_nnz_sorted)), lca_nnz_sorted)
    lca_nnz_xtick = range(0, lca_header['nf'] + 1, math.ceil(lca_header['nf']/8))
    lca_nnz_labels = ["{}".format(x) for x in lca_nnz_xtick]

    axs[0].set_xticks(lca_nnz_xtick)
    axs[0].set_xticklabels(lca_nnz_labels)

    figure_filename = str(Path(analysis_common.analysis_output, title_str).with_suffix('.png'))

    print('Saving LCA NNZ figure to {}...'.format(figure_filename))
    fh_nnz.savefig(figure_filename)

    print("Plotting LCA NNZ finished.")

    return lca_rank_index
