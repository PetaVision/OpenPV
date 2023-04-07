from openpv_analysis.AnalysisCommon import AnalysisCommon
from openpv_analysis.AnalysisSettings import AnalysisSettings
from openpv_analysis.pvtools.readpvpfile import readpvpfile
from openpv_analysis.pvtools.readpvpheader import readpvpheader
from openpv_analysis.plot_lca_nnz import plot_lca_nnz
from openpv_analysis.plot_energy import plot_energy
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def save_ata(analysis_common: AnalysisCommon, lca_rank_index = None):
    lca_path = os.path.join(analysis_common.checkpoint_path, 'V1_A.pvp')
    lca_dict = readpvpfile(lca_path)
    lca_header, lca_data = lca_dict['header'], lca_dict['values']

    i_lca = 1

    lca_rank_index = None

    ata_weights = None
    ata_rank_flag = lca_rank_index is not None
    ata_max_per_tableau = lca_header['nf']
    ata_num_tableau = math.ceil(lca_header['nf'] / ata_max_per_tableau)
    ata_num_per_tableau = min(ata_max_per_tableau, lca_header['nf'])
    ata_num_rows = math.ceil(math.sqrt(ata_num_per_tableau))
    ata_num_cols = math.ceil(ata_num_per_tableau / ata_num_rows)
    ata_min_val = np.tile(2.0 ** 31, (ata_num_tableau, ata_num_rows, ata_num_cols))
    ata_max_val = np.tile(-2.0 ** 31, (ata_num_tableau, ata_num_rows, ata_num_cols))

    ata_file_path = os.path.join(checkpoint_path, 'V1ToInputError_W.pvp')
    ata_header = readpvpheader(ata_file_path)
    ata_patch_size = [ata_header['nyp'], ata_header['nxp'], ata_header['nfp']]
    ata_dict = readpvpfile(ata_file_path)
    ata_data = ata_dict['values']
    ata_weights = ata_data[0, 0, :, :, :, :]

    for i_tableau in range(0, ata_num_tableau):
        for i_rank in range(i_tableau * ata_num_per_tableau, (i_tableau + 1) * ata_num_per_tableau):
            if ata_rank_flag:
                i_feature = lca_rank_index[i_rank]
            else:
                i_feature = i_rank

            ata_patch = np.squeeze(ata_weights[i_feature, :, :, :])
            ata_row, ata_col = np.unravel_index(i_rank - (i_tableau * ata_num_per_tableau),
                                                (ata_num_rows, ata_num_cols))
            ata_min_val[i_tableau, ata_row, ata_col] = min(np.min(ata_patch),
                                                            ata_min_val[i_tableau, ata_row, ata_col])
            ata_max_val[i_tableau, ata_row, ata_col] = max(np.max(ata_patch),
                                                            ata_max_val[i_tableau, ata_row, ata_col])


    for i_tableau in range(0, ata_num_tableau):

        file_name = "V{}ToInputError{}.png".format(i_lca, i_tableau)
        
        ata_array = np.zeros((ata_patch_size[0] * ata_num_rows, ata_patch_size[1] * ata_num_cols, 3))

        for i_rank in range(i_tableau * ata_num_per_tableau, (i_tableau + 1) * ata_num_per_tableau):
            if ata_rank_flag:
                i_feature = lca_rank_index[i_rank]
            else:
                i_feature = i_rank

            ata_patch = np.squeeze(ata_weights[i_feature, :, :, :])
            ata_row, ata_col = np.unravel_index(i_rank - (i_tableau * ata_num_per_tableau),
                                                (ata_num_rows, ata_num_cols))
            ata_abs_max = max(math.fabs(ata_max_val[i_tableau, ata_row, ata_col]),
                                math.fabs(ata_min_val[i_tableau, ata_row, ata_col]))

            if math.isclose(ata_abs_max, 0):
                ata_abs_max = 1

            ata_patch_normalized = ata_patch / ata_abs_max
            ata_uint8 = 127.5 * ata_patch_normalized + 127.500001
            ata_array[ata_row * ata_patch_size[0]:(ata_row + 1) * ata_patch_size[0],
            ata_col * ata_patch_size[1]:(ata_col + 1) * ata_patch_size[1]] = ata_uint8
        ata_array = ata_array.astype(np.uint8).repeat(5, axis=0).repeat(5, axis=1)
        save_path = os.path.join(analysis_common.analysis_output, file_name)
        plt.imsave(save_path, ata_array)



if __name__ == '__main__':
    # replace the paths
    output_path = '../output'
    checkpoint_path = os.path.join(output_path, 'Checkpoints/Checkpoint1250000')
    analysis_common = AnalysisCommon.initialize(AnalysisSettings(num_batch=8), 0, output_path, checkpoint_path)
    plot_lca_nnz(analysis_common)
    save_ata(analysis_common)
    plot_energy(analysis_common)
