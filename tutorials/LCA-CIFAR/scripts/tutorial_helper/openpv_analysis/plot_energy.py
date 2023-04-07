from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from .AnalysisCommon import AnalysisCommon


def plot_energy(analysis_common: AnalysisCommon):
    analysis_path = analysis_common.analysis_path
    analysis_output = analysis_common.analysis_output

    start_lca = analysis_common.analysis_settings.start_lca
    skip_lca = analysis_common.analysis_settings.skip_lca
    num_batch = analysis_common.analysis_settings.num_batch
    batch_skip = analysis_common.analysis_settings.batch_skip
    display_period = analysis_common.analysis_settings.display_period
    num_display_periods = analysis_common.analysis_settings.display_multiple

    energy_title_str = "energy{}_{}".format(start_lca, skip_lca)

    figure_energy, axis_energy = plt.subplots(1, 1)
    figure_energy.suptitle(energy_title_str)
    figure_energy.set_size_inches(analysis_common.analysis_settings.plot_size_in[0],
                                  analysis_common.analysis_settings.plot_size_in[1])

    objective_title_str = "objective{}_{}".format(start_lca, skip_lca)

    figure_objective, axis_objective = plt.subplots(1, 1)
    figure_objective.suptitle(objective_title_str)
    figure_objective.set_size_inches(analysis_common.analysis_settings.plot_size_in[0],
                                     analysis_common.analysis_settings.plot_size_in[1])

    if Path(analysis_path, "V1L1NormEnergyProbe_batchElement_0.txt").is_file():
        probe_title_str = "V1L1NormEnergyProbe_{}_{}".format(start_lca, skip_lca)
        figure_probe, axis_probe = plt.subplots(1, 1)
        figure_probe.suptitle(probe_title_str)
        figure_probe.set_size_inches(analysis_common.analysis_settings.plot_size_in[0],
                                     analysis_common.analysis_settings.plot_size_in[1])
    else:
        probe_title_str = None
        figure_probe = None
        axis_probe = None

    energy_count = None

    for i in range(0, num_batch, batch_skip):
        energy_file = Path(analysis_path, "V1EnergyProbe_batchElement_{}.txt".format(i))
        probe_file = Path(analysis_path, "V1L1NormEnergyProbe_batchElement_{}.txt".format(i))

        print("Analysing {}".format(str(energy_file)))

        if not energy_file.is_file():
            print("WARNING: No energy file found {}".format(str(energy_file)))
            continue

        with open(energy_file, 'r') as fp:
            energy_lines = list(line.strip() for line in fp.readlines() if
                                not (line.startswith('Probe_name') or line.startswith('time')))

        if probe_file.is_file():
            print("Analysing {}".format(str(probe_file)))

            with open(probe_file, 'r') as fp:
                probe_lines = list(line.strip() for line in fp.readlines() if
                                   not (line.startswith('Probe_name') or line.startswith('time')))

            probe_values = np.array([float(line.split(',')[3].strip()) for line in probe_lines])
        else:
            probe_values = None

        index_index = 1
        energy_index = 3

        if len(energy_lines[0].split(',')) == 3:
            index_index -= 1
            energy_index -= 1

        energy_indices = np.array([int(float(line.split(',')[index_index].strip())) for line in energy_lines])
        energy_values = np.array([float(line.split(',')[energy_index].strip()) for line in energy_lines])

        energy_display_indices = np.where(energy_indices % display_period == 0)[0]

        energies_dp = energy_values.take(energy_display_indices)
        objective = energy_values[energy_display_indices[-num_display_periods]:energy_display_indices[-1]]

        if not energy_count:
            energy_count = energy_display_indices.size

        axis_energy.plot(energies_dp, label='Batch {}'.format(i + 1))
        axis_energy.legend()

        axis_objective.plot(objective, label='Batch {}'.format(i + 1))
        axis_objective.legend()

        if probe_values is not None:
            probe_values = probe_values[energy_display_indices[-num_display_periods]:energy_display_indices[-1]]
            axis_probe.plot(probe_values, label='Batch {}'.format(i + 1))
            axis_probe.legend()

    figure_filename = str(Path(analysis_output, "{}_{}.png".format(energy_title_str, energy_count)))
    figure_energy.savefig(figure_filename)

    figure_filename = str(Path(analysis_output, "{}_{}.png".format(objective_title_str, energy_count)))
    figure_objective.savefig(figure_filename)

    if probe_title_str:
        figure_filename = str(Path(analysis_output, "{}_{}.png".format(probe_title_str, energy_count)))
        figure_probe.savefig(figure_filename)
