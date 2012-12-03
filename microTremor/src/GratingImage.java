import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

/**
 * 
 */

/**
 * @author garkenyon
 * 
 */
public class GratingImage {

	int numCones;
	int numPixels;
	double gratingOrientation = Math.PI;
	double gratingSpatialFreq = 20.0; // cycles per degree
	double gratingContrast = 1.0; // units of background luminance
	double gratingSigma = 2.0; // units of spatial period
	double gratingPhase = 0.0; // radians

	/**
	 * 
	 */
	public GratingImage() {
		numCones = 128;
		numPixels = 8;
	}

	public GratingImage(int num_cones, int num_pixels,
			double grating_orientation, double grating_spatial_freq,
			double grating_contrast, double grating_sigma, double grating_phase) {
		numCones = num_cones;
		numPixels = num_pixels;
		gratingOrientation = grating_orientation;
		gratingSpatialFreq = grating_spatial_freq;
		gratingContrast = grating_contrast;
		gratingSigma = grating_sigma;
		gratingPhase = grating_phase;
	}

	public final static DoubleMatrix2D getGratingImage(int num_cones,
			int num_pixels, final double grating_orientation,
			final double grating_spatial_freq, final double grating_contrast,
			final double grating_sigma,
			final double grating_phase) {
		final double background_luminance = RetinalConstants.backgroundLuminance;
		double microns_per_degree = RetinalConstants.micronsPerDegree;
		double cone_diameter = RetinalConstants.coneDiameter;
		double pixel_cone_ratio = num_pixels / num_cones;
		double pixel_diameter = cone_diameter / pixel_cone_ratio;
		double half_pixel_diameter = pixel_diameter / 2.0;
		double[] pixel_vals = new double[num_pixels];
		for (int i_pixel = 0; i_pixel < num_pixels; i_pixel++) {
			pixel_vals[i_pixel] = half_pixel_diameter + i_pixel
					* pixel_diameter / microns_per_degree; // in degrees
		}
		DoubleMatrix2D pixel_x = new DenseDoubleMatrix2D(num_pixels, num_pixels);
		DoubleMatrix2D pixel_y = new DenseDoubleMatrix2D(num_pixels, num_pixels);
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_pixels; j_col++) {
				pixel_x.set(j_col, i_row, pixel_vals[j_col]);
				pixel_y.set(j_col, i_row, pixel_vals[i_row]);
			}
		}
		final double x_center = num_pixels * pixel_diameter
				/ (microns_per_degree * 2);
		final double y_center = num_pixels * pixel_diameter
				/ (microns_per_degree * 2);
		DoubleMatrix2D dist_perpendicular = pixel_x.copy();
		dist_perpendicular.assign(pixel_y, new DoubleDoubleFunction() {

			@Override
			public double apply(double pixel_x_val, double pixel_y_val) {
				double y_prime = Math.cos(grating_orientation)
						* (pixel_y_val - y_center)
						- Math.sin(grating_orientation)
						* (pixel_x_val - x_center);
				return y_prime;
			}
		});
		DoubleMatrix2D dist_2D = pixel_x.copy();
		dist_2D.assign(pixel_y, new DoubleDoubleFunction() {

			@Override
			public double apply(double pixel_x_val, double pixel_y_val) {
				return (pixel_x_val - x_center) * (pixel_x_val - x_center)
						+ (pixel_y_val - y_center) * (pixel_y_val - y_center);
			}
		});
		DoubleMatrix2D grating_vals = dist_2D.copy();
		grating_vals.assign(dist_perpendicular, new DoubleDoubleFunction() {

			@Override
			public double apply(double dist_2D_val,
					double dist_perpendicular_val) {
				return (grating_contrast
						* background_luminance
						* Math.cos(2 * Math.PI * grating_spatial_freq
								* dist_perpendicular_val + grating_phase) * Math.exp((-0.5)
						* (grating_spatial_freq * dist_2D_val / grating_sigma)));
			}
		});
		return grating_vals;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
