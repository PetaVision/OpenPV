import math
from pathlib import Path
import numpy as np
from PIL import Image

from .AnalysisCommon import AnalysisCommon
from pvtools import readpvpfile


def plot_ata(analysis_common: AnalysisCommon, lca_rank_index=None):
    print("Generating ATA analysis images...")

    lca_header, lca_data = analysis_common.read_v1_a_file()

    i_lca = 1

    ata_weights = [None] * analysis_common.analysis_settings.temporal_patch_size
    ata_rank_flag = lca_rank_index is not None
    ata_max_per_tableau = lca_header['nf']
    ata_num_tableau = math.ceil(lca_header['nf'] / ata_max_per_tableau)
    ata_num_per_tableau = min(ata_max_per_tableau, lca_header['nf'])
    ata_num_rows = math.ceil(math.sqrt(ata_num_per_tableau))
    ata_num_cols = math.ceil(ata_num_per_tableau / ata_num_rows)
    ata_min_val = np.tile(2.0 ** 31, (ata_num_tableau, ata_num_rows, ata_num_cols))
    ata_max_val = np.tile(-2.0 ** 31, (ata_num_tableau, ata_num_rows, ata_num_cols))
    ata_patch_size = analysis_common.ata_patch_size

    movie_frames = []
    movie_ata_time = -1

    for i_frame in range(0, analysis_common.analysis_settings.temporal_patch_size):
        ata_filename = Path(analysis_common.checkpoint_path,
                            'V{}ToInputError_W.pvp'.format(i_lca))

        if ata_filename.is_file():
            print("Analysing {} [{}/{}]...".format(ata_filename, i_frame + 1,
                                                   analysis_common.analysis_settings.temporal_patch_size))
            ata_dict = readpvpfile(ata_filename)
            ata_data = ata_dict['values']
        else:
            raise Exception("ATA Filename {} does not exist.".format(str(ata_filename)))

        ata_weights[i_frame] = ata_data[0, 0, :, :, :, :]

        for i_tableau in range(0, ata_num_tableau):
            for i_rank in range(i_tableau * ata_num_per_tableau, (i_tableau + 1) * ata_num_per_tableau):
                if ata_rank_flag:
                    i_feature = lca_rank_index[i_rank]
                else:
                    i_feature = i_rank

                ata_patch = np.squeeze(ata_weights[i_frame][i_feature, :, :, :])
                ata_row, ata_col = np.unravel_index(i_rank - (i_tableau * ata_num_per_tableau),
                                                    (ata_num_rows, ata_num_cols))
                ata_min_val[i_tableau, ata_row, ata_col] = min(np.min(ata_patch),
                                                               ata_min_val[i_tableau, ata_row, ata_col])
                ata_max_val[i_tableau, ata_row, ata_col] = max(np.max(ata_patch),
                                                               ata_max_val[i_tableau, ata_row, ata_col])

    for i_frame in range(0, analysis_common.analysis_settings.temporal_patch_size):
        ata_filename = Path(analysis_common.checkpoint_path,
                            'V{}ToInputError_W.pvp'.format(i_lca))

        if ata_filename.is_file():
            print("Generating patch for {} [{}/{}]...".format(ata_filename, i_frame + 1,
                                                              analysis_common.analysis_settings.temporal_patch_size))
            ata_dict = readpvpfile(ata_filename)
            ata_data = ata_dict['values']
            ata_time = int(ata_dict['time'][0])

            if movie_ata_time < 0:
                movie_ata_time = ata_time
        else:
            raise Exception("ATA Filename {} does not exist.".format(str(ata_filename)))

        ata_weights[i_frame] = ata_data[0, 0, :, :, :, :]

        for i_tableau in range(0, ata_num_tableau):
            title_str = "V{}ToInput{}_{}_{}_{}".format(i_lca, i_frame,
                                                                analysis_common.analysis_settings.start_lca,
                                                                analysis_common.analysis_settings.skip_lca, ata_time,
                                                                i_tableau)

            ata_array = np.zeros((ata_patch_size[0] * ata_num_rows, ata_patch_size[1] * ata_num_cols))

            for i_rank in range(i_tableau * ata_num_per_tableau, (i_tableau + 1) * ata_num_per_tableau):
                if ata_rank_flag:
                    i_feature = lca_rank_index[i_rank]
                else:
                    i_feature = i_rank

                ata_patch = np.squeeze(ata_weights[i_frame][i_feature, :, :, :])
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

            ata_array_image = Image.fromarray(ata_array.astype(dtype=np.uint8))

            if not math.isclose(analysis_common.analysis_settings.plot_ata_scale, 1.0):
                scale = analysis_common.analysis_settings.plot_ata_scale

                ata_img_size = ata_array_image.size
                ata_img_new_size = (int(ata_img_size[0] * scale), int(ata_img_size[1] * scale))

                print("Resizing ATA image [{}, {}] => [{}, {}]".format(ata_img_size[0], ata_img_size[1],
                                                                       ata_img_new_size[0], ata_img_new_size[1]))
                ata_array_image = ata_array_image.resize(ata_img_new_size, Image.ANTIALIAS)

            movie_frames.append(ata_array_image)

            image_path = Path(analysis_common.analysis_output, title_str).with_suffix('.png')
            print("Saving ATA patch image {}...".format(str(image_path)))

            ata_array_image.save(str(image_path))

    movie_str = "S{}OracleToFrame_{}_{}_{}_movie.gif".format(i_lca,
                                                             analysis_common.analysis_settings.start_lca,
                                                             analysis_common.analysis_settings.skip_lca,
                                                             movie_ata_time)

    print("Generating Animated GIF {}...".format(str(Path(analysis_common.analysis_output, movie_str))))

    movie_frames[0].save(str(Path(analysis_common.analysis_output, movie_str)), format='GIF',
                         append_images=movie_frames[slice(1, -1)], save_all=True, duration=200, loop=0)

    print("ATA generation finished.")
