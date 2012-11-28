import java.util.Random;
import java.lang.Math;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import cern.colt.function.tdcomplex.DComplexDComplexDComplexFunction;
import cern.colt.function.tdcomplex.DComplexDComplexFunction;
import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.*;
import cern.colt.matrix.tdcomplex.DComplexFactory1D;
import cern.colt.matrix.tdcomplex.DComplexFactory2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.jet.math.Functions;
import cern.jet.math.tdcomplex.DComplexFunctions;

import edu.emory.mathcs.*;

import flanagan.integration.*;

/**
 * @author garkenyon
 * 
 */
public class OcularTremor {

	long numSteps;
	double deltaT;
	Random tremorGenerator;

	// double background_luminance = 100.0; // trolands

	OcularTremor(Random generator) {
		super();
		tremorGenerator = generator;
	}

	// we compute the random tremor time series only once per image so we don't pre-allocate temporary arrays
	DenseDComplexMatrix1D getTimeSeries(double delta_t, long num_steps){
		numSteps = num_steps;
		deltaT = delta_t;
		double peakDrift = RetinalConstants.peakDrift;
		double peakTremor = RetinalConstants.peakTremor;
		double freqTremor = RetinalConstants.freqTremor;
		double sigmaTremor = RetinalConstants.sigmaTremor;

		// compute tremor time series
		double[] time_steps = new double[(int) numSteps]; // 0 : delta_t :
		// (num_steps-1)*delta_t;
		for (int i_step = 0; i_step < numSteps; i_step++) {
			time_steps[i_step] = i_step * deltaT;
		}
		double[] freq_vals = new double[(int) num_steps]; // freq_vals = (
		// 0:(num_steps-1) ) ./ (
		// delta_t * num_steps );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			freq_vals[i_step] = i_step / (deltaT * numSteps);
		}
		double freq_drift = 1.0; // Hz
		DenseDoubleMatrix1D amp_low = new DenseDoubleMatrix1D((int) numSteps); // peak_drift ./ ( 1 + (
		// freq_vals ./ freq_drift )
		// );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double amp_low_val = peakDrift
					/ (1 + (freq_vals[i_step] / freq_drift));
			amp_low.set(i_step, amp_low_val);
		}
		DenseDoubleMatrix1D amp_high = new DenseDoubleMatrix1D((int) numSteps); // peak_tremor .* exp( -(1/2)
		// .* ( ( freq_vals -
		// freq_tremor ).^2 ) ./ (
		// sigma_tremor^2 ) );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double amp_high_val = peakTremor
					* Math.exp(-0.5
							*Math.pow((freq_vals[i_step] - freqTremor) / sigmaTremor, 2.0));
			amp_high.set(i_step, amp_high_val);
		}
		DenseDComplexMatrix1D tremor_phase = new DenseDComplexMatrix1D((int) numSteps);
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double ran_phase = 2.0*Math.PI*tremorGenerator.nextDouble();
			tremor_phase.set(i_step, Math.cos(ran_phase), Math.sin(ran_phase));
		}

		//double[] tremor_amp = new double[(int) numSteps]; // x_amp * ( amp_low + amp_high ) .* tremor_phase(1,:);
		//for (int i_step = 0; i_step < numSteps; i_step++) {
			//tremor_amp[i_step] = (amp_low[i_step] + amp_high[i_step]) * tremor_phase[i_step];
		//}
		DenseDComplexMatrix1D amp_low_dcomplex = new DenseDComplexMatrix1D((int) numSteps);  // initialize with low frequency "drift" amp
		amp_low_dcomplex.assignReal(amp_low);
		amp_low_dcomplex.assignImaginary(new DenseDoubleMatrix1D((int) numSteps));
		DenseDComplexMatrix1D amp_high_dcomplex = new DenseDComplexMatrix1D((int) numSteps);  // initialize with low frequency "drift" amp
		amp_high_dcomplex.assignReal(amp_high);
		amp_high_dcomplex.assignImaginary(new DenseDoubleMatrix1D((int) numSteps));
		DenseDComplexMatrix1D fft_tremor = new DenseDComplexMatrix1D(amp_low_dcomplex.toArray());
		fft_tremor.assign(amp_high_dcomplex, DComplexFunctions.plus );		
		fft_tremor.assign(amp_high_dcomplex, DComplexFunctions.mult );		
		fft_tremor.ifft(true); // 2D Fourier Transform in place,
											// arg=true specifies scaling is to
											// be performed (see Parallel Colt
											// API)
		return fft_tremor;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {

	} // end main()

} // end Class TremorMain

