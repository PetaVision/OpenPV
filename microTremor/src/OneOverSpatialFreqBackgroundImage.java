import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.image.BandedSampleModel;
import java.awt.image.ColorModel;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import javax.media.jai.TiledImage;
import javax.media.jai.PlanarImage;
import javax.swing.JFrame;
import javax.swing.JScrollPane;

import com.sun.media.jai.widget.DisplayJAI;

import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdcomplex.DComplexFactory2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplexFunctions;
import cern.jet.math.tdouble.DoubleFunctions;

@SuppressWarnings("restriction")
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

	public DoubleMatrix2D getBackgroundImage() {
		return getBackgroundImage(numCones, numPixels);
	}

	public final static DenseDoubleMatrix2D getBackgroundImage(int num_cones,
			int num_pixels) {
		// cern.jet.math.Functions F = cern.jet.math.Functions.functions; //
		// naming shortcut (alias) saves some keystrokes:
		//final double background_luminance = RetinalConstants.backgroundLuminance;
		double microns_per_degree = RetinalConstants.micronsPerDegree;
		double cone_diameter = RetinalConstants.coneDiameter;
		double pixel_cone_ratio = (double) num_pixels / (double) num_cones;
		double pixel_diameter = cone_diameter / pixel_cone_ratio;
		double sp_peak_bkgrnd = 1.0; // 0.75 * 1.0 * background_luminance; // trolands
		double[] sp_freqs = new double[num_pixels];
		// cycles per micron -> cycles per degree
		for (int i_freq = 0; i_freq < num_pixels; i_freq++) {
			sp_freqs[i_freq] = i_freq * microns_per_degree
					/ (num_pixels * pixel_diameter);
		}
		double sp_freq_bkgrnd = 2.0; // cycles per degree
		DenseDoubleMatrix2D sp_freq_2D = new DenseDoubleMatrix2D(num_pixels,
				num_pixels);
		DenseDoubleMatrix2D sp_freq_y = new DenseDoubleMatrix2D(num_pixels,
				num_pixels);
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				sp_freq_2D.set(i_row, j_col, sp_freqs[j_col]);
				sp_freq_y.set(i_row, j_col, sp_freqs[i_row]);
			}
		}
		sp_freq_2D.assign(DoubleFunctions.square);
		sp_freq_y.assign(DoubleFunctions.square);
		// DenseDoubleMatrix2D sp_freq_2D = (DenseDoubleMatrix2D)
		// sp_freq_x.copy();
		sp_freq_2D.assign(sp_freq_y, DoubleFunctions.plus);
		sp_freq_2D.assign(DoubleFunctions.sqrt);
		DenseDoubleMatrix2D sp_freq_scale2D = new DenseDoubleMatrix2D(
				num_pixels, num_pixels);
		sp_freq_scale2D.assign(sp_freq_bkgrnd);
		sp_freq_2D.assign(sp_freq_scale2D, DoubleFunctions.div);
		DenseDoubleMatrix2D sp_amp_bkgrnd_scale = new DenseDoubleMatrix2D(
				num_pixels, num_pixels);
		sp_amp_bkgrnd_scale.assign(sp_peak_bkgrnd);
		DenseDoubleMatrix2D sp_amp_bkgrnd2D = new DenseDoubleMatrix2D(
				num_pixels, num_pixels);
		sp_amp_bkgrnd2D.assign(1.0);
		sp_amp_bkgrnd2D.assign(sp_freq_2D, DoubleFunctions.plus);
		sp_amp_bkgrnd2D.assign(DoubleFunctions.inv);
		sp_amp_bkgrnd2D.assign(sp_amp_bkgrnd_scale, DoubleFunctions.mult);
		sp_amp_bkgrnd2D.set(0, 0, 0.0); // use background_luminance for zero
										// frequency amp;
		DComplexMatrix2D sp_phase_bkgrnd = DComplexFactory2D.dense.make(
				num_pixels, num_pixels);
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				double ran_phase = 2 * Math.PI * Math.random();
				sp_phase_bkgrnd.set(i_row, j_col, Math.cos(ran_phase),
						Math.sin(ran_phase));
			}
		}
		DenseDComplexMatrix2D sp_bkgrnd2D = (DenseDComplexMatrix2D) DComplexFactory2D.dense
				.make(num_pixels, num_pixels); // should initialize to zeros
		sp_bkgrnd2D.assignReal(sp_amp_bkgrnd2D);
		sp_bkgrnd2D.assign(sp_phase_bkgrnd, DComplexFunctions.mult);
		sp_bkgrnd2D.ifft2(false); // 2D Fourier Transform in place,
		//sp_amp_bkgrnd_scale.assign(num_pixels * num_pixels);
		DenseDoubleMatrix2D one_over_f_bkgrnd = (DenseDoubleMatrix2D) sp_bkgrnd2D
				.getRealPart();

		// scale 1/f background between 0 : 255
		one_over_f_bkgrnd.assign(ByteArray.convertDoubleToByte(one_over_f_bkgrnd.elements()));
		
		// draw 1/f random background image
		ByteArray.draw(one_over_f_bkgrnd, num_pixels, num_pixels, "1/f background", null);
		return one_over_f_bkgrnd;

	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
