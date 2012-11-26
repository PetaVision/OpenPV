import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdcomplex.DComplexFactory2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.Functions;

public class OneOverSpatialFreqBackgroundImage {

	int numCones;
	int numPixels;

	public OneOverSpatialFreqBackgroundImage() {
		numCones = 128;
		numPixels = 8;
	}
	
	public OneOverSpatialFreqBackgroundImage(int num_cones, int num_pixels) {
		numCones = num_cones;
		numPixels = num_pixels;
	}
	
	public DoubleMatrix2D getBackgroundImage(){
		return getBackgroundImage(numCones, numPixels);
	}
	
	public final static DoubleMatrix2D getBackgroundImage(int num_cones, int num_pixels) {
		cern.jet.math.Functions F = cern.jet.math.Functions.functions; // naming shortcut (alias) saves some keystrokes:
		final double background_luminance = RetinalConstants.backgroundLuminance;
		double microns_per_degree = RetinalConstants.micronsPerDegree;
		double cone_diameter = RetinalConstants.coneDiameter;
		double pixel_cone_ratio = (double) num_pixels / (double) num_cones;
		double pixel_diameter = cone_diameter / pixel_cone_ratio;
		double sp_peak_bkgrnd = 0.75 * 1.0 * background_luminance; // trolands
		double[] sp_freqs = new double[num_pixels]; // = (0:(num_pixels-1)) / (
													// num_cones * cone_diameter
													// ); //cycles per micron
		for (int i_freq = 0; i_freq < num_pixels; i_freq++) {
			sp_freqs[i_freq] = i_freq * microns_per_degree
					/ (num_cones * pixel_diameter);
		}
		// cycles per micron -> cycles per degree
		double[][] sp_freq_x = new double[num_pixels][num_pixels];
		double[][] sp_freq_y = new double[num_pixels][num_pixels];
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				sp_freq_x[i_row][j_col] = sp_freqs[j_col];
				sp_freq_y[i_row][j_col] = sp_freqs[i_row];
			}
		}
		double sp_freq_bkgrnd;
		sp_freq_bkgrnd = 2.0; // cycles per degree
		double[][] sp_amp_bkgrnd = new double[num_pixels][num_pixels];
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				sp_amp_bkgrnd[i_row][j_col] = sp_peak_bkgrnd
						/ (1 + Math.sqrt(Math.pow(sp_freq_x[i_row][j_col], 2)
								+ Math.pow(sp_freq_y[i_row][j_col], 2))
								/ sp_freq_bkgrnd);
			}
		}
		sp_amp_bkgrnd[0][0] = 0.0; // use background_luminance;
		DComplexMatrix2D sp_phase_bkgrnd = DComplexFactory2D.dense.make(
				num_pixels, num_pixels);
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				double ran_phase = 2 * Math.PI * Math.random();
				sp_phase_bkgrnd.set(i_row, j_col, Math.cos(ran_phase),
						Math.sin(ran_phase));
			}
		}
		DComplexMatrix2D sp_bkgrnd = DComplexFactory2D.dense.make(num_pixels,
				num_pixels);
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				sp_bkgrnd.set(i_row, j_col,
						sp_phase_bkgrnd.getRealPart().get(i_row, j_col)
								* sp_amp_bkgrnd[i_row][j_col], sp_phase_bkgrnd
								.getImaginaryPart().get(i_row, j_col)
								* sp_amp_bkgrnd[i_row][j_col]);
			}
		}
		DenseDComplexMatrix2D ifft2_bckgrnd_1_over_f = new DenseDComplexMatrix2D(
				num_pixels, num_pixels);
		ifft2_bckgrnd_1_over_f.assignReal(sp_bkgrnd.getRealPart());
		ifft2_bckgrnd_1_over_f.assignImaginary(sp_bkgrnd.getImaginaryPart());
		ifft2_bckgrnd_1_over_f.ifft2(true); // 2D Fourier Transform in place,
											// arg=true specifies scaling is to
											// be performed (see Parallel Colt
											// API)
		DenseDoubleMatrix2D bckgrnd_1_over_f = (DenseDoubleMatrix2D) ifft2_bckgrnd_1_over_f.getRealPart();
		final double mean_bckgrnd = bckgrnd_1_over_f.zSum();
		bckgrnd_1_over_f.assign(new DoubleFunction() {
			
			@Override
			public double apply(double bckgrnd_val) {
				bckgrnd_val = bckgrnd_val - mean_bckgrnd + background_luminance;
				return (bckgrnd_val > 0) ? bckgrnd_val : 0.0;
			}
		});
		
//		bckgrnd_1_over_f.assign((DoubleFunction) Functions.minus(mean_bckgrnd));
//		bckgrnd_1_over_f.assign((DoubleFunction) Functions.plus(background_luminance));
//		bckgrnd_1_over_f.assign(
//				bckgrnd_1_over_f.copy().assign((DoubleFunction) Functions.greater(0)),
//				(DoubleDoubleFunction) Functions.mult);
		return bckgrnd_1_over_f;

	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
