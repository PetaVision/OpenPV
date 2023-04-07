
import numpy as np
from pathlib import Path

from .AnalysisSettings import AnalysisSettings

from pvtools.readpvpheader import readpvpheader
from pvtools.readpvpfile import readpvpfile


class AnalysisCommon(object):

    def __init__(self, analysis_settings: AnalysisSettings, lca_index):

        self.analysis_settings = analysis_settings

        self.v_thresh = np.zeros((1, analysis_settings.lca_num_layers))
        self.v_thresh_bins = [None] * analysis_settings.lca_num_layers

        for i in range(0, analysis_settings.lca_num_layers):
            self.v_thresh[i] = analysis_settings.v_thresh_base * (analysis_settings.v_thresh_scale ** i)
            self.v_thresh_bins[i] = self.v_thresh[i] * (np.arange(0, 5.5, 0.25, dtype=np.float) - 0.125)

        self.original_rank_index = []
        self.lca_index = lca_index

        self.analysis_path = None
        self.checkpoint_path = None

        self.lca_filename = None
        self.lca_header = None
        self.min_num_lca = None

        self.ata_filename = None
        self.ata_header = None
        self.ata_patch_size = None

        self.image_filename = None
        self.image_header = None

        self.analysis_output = None

    @staticmethod
    def initialize(analysis_settings, lca_index, analysis_path, checkpoint_path):

        lca_file_path = Path(analysis_path, 'V{}.pvp'.format(lca_index + 1))

        if not lca_file_path.is_file():
            lca_file_path = Path(checkpoint_path, 'V{}_A.pvp'.format(lca_index + 1))

        if not lca_file_path.is_file():
            print('Could not locate LCA file: {}'.format(lca_file_path))
            return None

        lca_header = readpvpheader(str(lca_file_path))
        min_num_lca = min([analysis_settings.min_num_lca, lca_header['nbands']])

        ata_filename = Path(checkpoint_path, 'V1ToInputError_W.pvp')

        if not ata_filename.is_file():
            print('ATA file does not exist: {}'.format(ata_filename))
            return None

        ata_header = readpvpheader(str(ata_filename))
        ata_patch_size = [ata_header['nyp'], ata_header['nxp'], ata_header['nfp']]

        image_filename = Path(checkpoint_path, 'Input_A.pvp')

        if not image_filename.is_file():
            print('Image file does not exist: {}'.format(image_filename))
            return None

        image_header = readpvpheader(str(image_filename))

        analysis_common = AnalysisCommon(analysis_settings, lca_index)

        analysis_common.analysis_path = analysis_path
        analysis_common.checkpoint_path = checkpoint_path

        analysis_common.lca_filename = str(lca_file_path)
        analysis_common.lca_header = lca_header
        analysis_common.min_num_lca = min_num_lca
        
        analysis_common.ata_filename = str(ata_filename)
        analysis_common.ata_header = ata_header
        analysis_common.ata_patch_size = ata_patch_size
        
        analysis_common.image_filename = str(image_filename)
        analysis_common.image_header = image_header

        analysis_output = Path(analysis_path, 'analysis_{}'.format(Path(checkpoint_path).name))
        analysis_output.mkdir(exist_ok=True)

        analysis_common.analysis_output = str(analysis_output)

        return analysis_common

    def read_v1_a_file(self):
        lca_filename = str(Path(self.checkpoint_path, 'V1_A.pvp'))

        lca_dict = readpvpfile(lca_filename)

        return lca_dict['header'], lca_dict['values']

        


