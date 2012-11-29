import java.util.Random;

import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.jet.math.tdcomplex.DComplexFunctions;

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
			freq_vals[i_step] = 1000.0 * i_step / (deltaT * numSteps);
		}
		double freq_drift = 1.0; // Hz
		DenseDComplexMatrix1D amp_low = new DenseDComplexMatrix1D((int) numSteps); // peak_drift ./ ( 1 + (
		// freq_vals ./ freq_drift )
		// );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double[] amp_low_val = new double[2];
			amp_low_val[0] = peakDrift
					/ (1 + (freq_vals[i_step] / freq_drift));
			amp_low.set(i_step, amp_low_val);
		}
		DenseDComplexMatrix1D amp_high = new DenseDComplexMatrix1D((int) numSteps); // peak_tremor .* exp( -(1/2)
		// .* ( ( freq_vals -
		// freq_tremor ).^2 ) ./ (
		// sigma_tremor^2 ) );
		for (int i_step = 0; i_step < numSteps; i_step++) {
			double[] amp_high_val = new double[2];
			amp_high_val[0] = peakTremor
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
		DenseDComplexMatrix1D fft_tremor = new DenseDComplexMatrix1D(amp_low.toArray());
		fft_tremor.assign(amp_high, DComplexFunctions.plus );		
		fft_tremor.assign(tremor_phase, DComplexFunctions.mult );		
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

