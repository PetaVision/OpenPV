import java.util.Random;

import cern.colt.function.tdcomplex.DComplexDComplexFunction;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplexFunctions;
import flanagan.plot.PlotGraph;

/**
 * @author garkenyon
 * 
 */
public class OcularTremor {

	int numSteps;
	double deltaT;
	Random tremorGenerator;

	// double background_luminance = 100.0; // trolands

	OcularTremor(Random generator) {
		super();
		tremorGenerator = generator;
	}

	// we compute the random tremor time series only once per image so we don't
	// pre-allocate temporary arrays
	DenseDComplexMatrix1D getTimeSeries(double delta_t, int num_steps) {
		numSteps = num_steps;
		deltaT = delta_t;
		double degrees_per_cone = RetinalConstants.coneDiameter
				/ RetinalConstants.micronsPerDegree;
		double peakDrift = RetinalConstants.peakDrift / degrees_per_cone;
		double peakTremor = RetinalConstants.peakTremor / degrees_per_cone;
		double freqTremor = RetinalConstants.freqTremor;
		double sigmaTremor = RetinalConstants.sigmaTremor;

		// compute tremor time series
		double[] time_steps = new double[(int) numSteps]; // 0 : delta_t :
		// (num_steps-1)*delta_t;
		for (int i_step = 0; i_step < numSteps; i_step++) {
			time_steps[i_step] = i_step * deltaT;
		}
		double[] freq_vals = new double[(int) num_steps];
		for (int i_step = 0; i_step < numSteps; i_step++) {
			freq_vals[i_step] = 1000.0 * i_step / (deltaT * numSteps);
		}
		int max_freq_index = (int) Math
				.ceil(140.0 * deltaT * numSteps / 1000.0);
		double freq_drift = 1.0; // Hz
		DenseDComplexMatrix1D amp_low = new DenseDComplexMatrix1D(
				(int) numSteps);
		// );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double[] amp_low_val = new double[2];
			amp_low_val[0] = peakDrift / (1 + (freq_vals[i_step] / freq_drift));
			amp_low.set(i_step, amp_low_val);
		}
		DenseDComplexMatrix1D amp_high = new DenseDComplexMatrix1D(
				(int) numSteps);
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double[] amp_high_val = new double[2];
			amp_high_val[0] = peakTremor
					* Math.exp(-0.5
							* Math.pow((freq_vals[i_step] - freqTremor)
									/ sigmaTremor, 2.0));
			amp_high.set(i_step, amp_high_val);
		}
		DenseDComplexMatrix1D tremor_phase = new DenseDComplexMatrix1D(
				(int) numSteps);
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double ran_phase = 2.0 * Math.PI * tremorGenerator.nextDouble();
			tremor_phase.set(i_step, Math.cos(ran_phase), Math.sin(ran_phase));
		}
		DenseDComplexMatrix1D micro_tremor = new DenseDComplexMatrix1D(
				amp_low.toArray());
		micro_tremor.assign(amp_high, DComplexFunctions.plus);

		// plot Power spectrum
		DenseDoubleMatrix1D tremor_power = (DenseDoubleMatrix1D) micro_tremor
				.getRealPart().copy();
		PlotGraph tremorPowerPlot2D = new PlotGraph(new DenseDoubleMatrix1D(
				freq_vals).viewPart(1, max_freq_index).toArray(), tremor_power
				.viewPart(1, max_freq_index).toArray());
		tremorPowerPlot2D.setGraphTitle("Tremor Power Spectrum");
		tremorPowerPlot2D.setXaxisLegend("Temporal Frequency");
		tremorPowerPlot2D.setXaxisUnitsName("Hz");
		tremorPowerPlot2D.setLine(1);
		tremorPowerPlot2D.plot();
		tremorPowerPlot2D.setCloseChoice(2);

		micro_tremor.assign(tremor_phase, DComplexFunctions.mult);
		micro_tremor.ifft(false); // 1D Fourier Transform in place, arg=true
									// specifies scaling is to be performed (see
									// Parallel Colt API)
		double[] micro_tremor_sum = micro_tremor.zSum(); //
		DenseDComplexMatrix1D mean_tremor = new DenseDComplexMatrix1D(
				(int) num_steps);
		mean_tremor.assign(micro_tremor_sum[0] / num_steps, micro_tremor_sum[1]
				/ num_steps);
		micro_tremor.assign(mean_tremor, DComplexFunctions.minus);

		// plot vertical tremor motion vs time step: amplitude in dimensionless
		// units (cone diameters)
		PlotGraph tremorVerticalPlot2D = new PlotGraph(time_steps, micro_tremor
				.getImaginaryPart().toArray());
		tremorVerticalPlot2D.setLine(3);
		tremorVerticalPlot2D
				.setGraphTitle("Vertical Tremor (in cone diameters)");
		tremorVerticalPlot2D.setXaxisLegend("Time (msec)");
		tremorVerticalPlot2D.setYaxisLegend("displacement");
		tremorVerticalPlot2D.plot();
		tremorVerticalPlot2D.setCloseChoice(2);

		// plot horizontal tremor motion vs time step: amplitude in
		// dimensionless units (cone diameters)
		PlotGraph tremorHorizontalPlot2D = new PlotGraph(time_steps,
				micro_tremor.getRealPart().toArray());
		tremorHorizontalPlot2D.setLine(3);
		tremorHorizontalPlot2D
				.setGraphTitle("Horizontal Tremor (in cone diameters)");
		tremorHorizontalPlot2D.setXaxisLegend("Time (msec)");
		tremorHorizontalPlot2D.setYaxisLegend("displacement");
		tremorHorizontalPlot2D.plot();
		tremorHorizontalPlot2D.setCloseChoice(2);

		return micro_tremor; // in cone diameters
	}

	public static DenseDoubleMatrix2D jitterImageUsingTremor(
			DenseDoubleMatrix2D input_matrix2D,
			DenseDComplexMatrix1D micro_tremor, int time_step,
			double pixel_cone_ratio) {
		double[] tremor_val = micro_tremor.get(time_step);
		tremor_val[0] *= pixel_cone_ratio;
		tremor_val[1] *= pixel_cone_ratio;
		int[] row_permute = new int[input_matrix2D.rows()];
		int row_shift = (int) (tremor_val[0] >= 0 ? input_matrix2D.rows()
				- tremor_val[0] : -tremor_val[0]);
		for (int i_row = 0; i_row < input_matrix2D.rows(); i_row++) {
			int i_shift_row = (i_row + row_shift) % input_matrix2D.rows();
			row_permute[i_row] = i_shift_row;
		}
		int[] col_permute = new int[input_matrix2D.columns()];
		int col_shift = (int) (tremor_val[1] >= 0 ? input_matrix2D.columns()
				- tremor_val[1] : -tremor_val[1]);
		for (int i_col = 0; i_col < input_matrix2D.columns(); i_col++) {
			int i_shift_col = (i_col + col_shift) % input_matrix2D.columns();
			col_permute[i_col] = i_shift_col;
		}
		DenseDoubleMatrix2D jittered_matrix2D = (DenseDoubleMatrix2D) input_matrix2D.copy();
		jittered_matrix2D = (DenseDoubleMatrix2D) input_matrix2D.viewSelection(row_permute, null).copy();
		jittered_matrix2D = (DenseDoubleMatrix2D) jittered_matrix2D.viewSelection(null, col_permute).copy();
		return jittered_matrix2D;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	} // end main()

} // end Class TremorMain

