

class AnalysisSettings(object):

    def __init__(self, num_batch=16):
        self.num_batch = num_batch
        self.num_batch_array = [None] * self.num_batch
        self.skip_lca = 1
        self.start_lca = 1

        self.display_multiple = 4
        self.display_period = self.display_multiple * 100
        self.temporal_patch_size = 1

        self.lca_num_layers = 1
        self.min_num_lca = 2**31-1

        self.v_thresh_base = 0.10
        self.v_thresh_scale = 1.0

        self.num_display_periods = 10
        self.batch_skip = 1

        self.dca_num_layers = 1
        self.dca_num_rows = 3

        self.plot_size_in = (18.5, 10.5)

        self.plot_ata_scale = 1.0

        self.plot_lca_nnz = True
        self.plot_ata = True
        self.plot_energy = True
        self.plot_image_recon = True
        self.plot_image_recon_subplot = False
