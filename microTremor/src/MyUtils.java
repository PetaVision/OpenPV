import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.image.BandedSampleModel;
import java.awt.image.ColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.SampleModel;

import javax.media.jai.PlanarImage;
import javax.media.jai.TiledImage;
import javax.swing.JFrame;
import javax.swing.JScrollPane;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

import com.sun.media.jai.widget.DisplayJAI;

@SuppressWarnings("restriction")
public class MyUtils {

	@SuppressWarnings("restriction")
	public static PlanarImage draw(DenseDoubleMatrix2D matrix_2D, int num_rows,
			int num_cols, String title_string, JFrame frame) {

		PlanarImage image_snap = MyUtils.mat2image(matrix_2D);
		MyUtils.display(image_snap, title_string, frame);
		return image_snap;
	}

	public static JFrame display(PlanarImage image_snap, String title_string,
			JFrame frame) {
		if (frame == null) {
			frame = new JFrame();
		}
		frame.setTitle(title_string);
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setSize(image_snap.getHeight(), image_snap.getWidth());
		frame.setVisible(true);
		return display(image_snap, frame);
	}

	public static JFrame display(PlanarImage image_snap, JFrame frame) {
		Container contentPane = frame.getContentPane();
		contentPane.setLayout(new BorderLayout());
		DisplayJAI image_display = new DisplayJAI(image_snap);
		contentPane.add(new JScrollPane(image_display), BorderLayout.CENTER);
		frame.setVisible(true);
		return frame;
	}

	public static double[] convertDoubleToByte(double[] double_vals) {
		DenseDoubleMatrix1D double_matrix1D = new DenseDoubleMatrix1D(
				double_vals);
		final double[] max_double_matrix1D = double_matrix1D.getMaxLocation();
		final double[] min_double_matrix1D = double_matrix1D.getMinLocation();
		final double range_double_matrix1D = max_double_matrix1D[0]
				- min_double_matrix1D[0] > 0 ? max_double_matrix1D[0]
				- min_double_matrix1D[0] : 255;
		double_matrix1D.assign(new DoubleFunction() {
			@Override
			public double apply(double double_val) {
				return Math.round(0.4999 + 255
						* (double_val - min_double_matrix1D[0])
						/ range_double_matrix1D);
			}
		});

		return double_matrix1D.elements();
	}

	public static PlanarImage mat2image(DenseDoubleMatrix2D matrix_2D) {
		matrix_2D.assign(MyUtils.convertDoubleToByte(matrix_2D.elements()));
		int num_rows = matrix_2D.rows();
		int num_cols = matrix_2D.columns();
		int num_pixels = num_rows * num_cols;
		byte[] image_data = new byte[num_pixels]; // Image data
		// array.
		double[] image_double = matrix_2D.elements();
		for (int i_pixel = 0; i_pixel < num_pixels; i_pixel++) {
			image_data[i_pixel] = (byte) image_double[i_pixel];
		}
		DataBufferByte dbuffer = new DataBufferByte(image_data, num_pixels);
		SampleModel sampleModel = new BandedSampleModel(DataBuffer.TYPE_BYTE,
				num_rows, num_cols, 1);
		ColorModel colorModel = PlanarImage.createColorModel(sampleModel);
		Raster raster = java.awt.image.Raster.createWritableRaster(sampleModel,
				dbuffer, null);
		TiledImage tiledImage = new TiledImage(0, 0, num_rows, num_cols, 0, 0,
				sampleModel, colorModel);
		tiledImage.setData(raster);
		PlanarImage image_snap = tiledImage.createSnapshot();
		return image_snap;
	}

	public static DenseDoubleMatrix2D downsample(DenseDoubleMatrix2D matrix_2D,
			DenseDoubleMatrix2D downsample_kernel) {
		DenseDoubleAlgebra dbl_algebra = new DenseDoubleAlgebra();
		DenseDoubleMatrix2D downsampled_matrix2D = (DenseDoubleMatrix2D) dbl_algebra.mult(downsample_kernel,
				dbl_algebra.mult(matrix_2D, downsample_kernel.viewDice().copy())).copy();
		return downsampled_matrix2D;
	}

	// kernel for downsampling 2D matrix by taking a non-overlapping local
	// average around each destination element
	public static DenseDoubleMatrix2D getDownsampleKernel(int size_pre,
			int size_post) {
		DenseDoubleMatrix2D downsample_kernel = new DenseDoubleMatrix2D(
				size_post, size_pre);
		double[][] kernel_vals = new double[size_post][size_pre];
		double ratio_pre_to_post = (double) size_pre / (double) size_post;
		for (int i_post = 0; i_post < size_post; i_post++) {
			double scaled_post = ratio_pre_to_post * (i_post + 0.5) - 0.5;
			for (int i_pre = 0; i_pre < size_pre; i_pre++) {
				double dist_abs = Math.abs(scaled_post - i_pre);
				kernel_vals[i_post][i_pre] = (dist_abs <= 0.5 * ratio_pre_to_post) ? (1 / ratio_pre_to_post)
						: 0.0;
			}
		}
		downsample_kernel.assign(kernel_vals);
		return downsample_kernel;
	}

	// converts a PlanerImage into a DoubleMatrix2D
	public static DenseDoubleMatrix2D image2mat(PlanarImage planar_image) {
		int num_rows = planar_image.getHeight();
		int num_cols = planar_image.getWidth();
		int num_pixels = num_rows * num_cols;
		DenseDoubleMatrix2D matrix_2D = new DenseDoubleMatrix2D(num_rows,
				num_cols);
		DataBuffer data_buffer = planar_image.getData().getDataBuffer();
		int i_pixel = 0;
		for (int i_row = 0; i_row < num_pixels; i_row++) {
			for (int j_col = 0; j_col < num_cols; j_col++) {
				matrix_2D.set(i_row, j_col,
						data_buffer.getElemDouble(i_pixel++));
			}
		}
		return matrix_2D;
	}

	// returns matrix with same dimensions as pattern_2D that equals baseline_val at center of pattern_2D range and varies by an amount
	// set by contrast_val
	public static DenseDoubleMatrix2D addScaled(
			DenseDoubleMatrix2D pattern_2D,
			final double baseline_val,
			final double contrast_val) {
		DenseDoubleMatrix2D baseline_2D = new DenseDoubleMatrix2D(pattern_2D.rows(), pattern_2D.columns());
		baseline_2D.assign(baseline_val);
		DenseDoubleMatrix2D contrast_2D = new DenseDoubleMatrix2D(pattern_2D.rows(), pattern_2D.columns());
		contrast_2D.assign(contrast_val);
		final double[] max_pattern = pattern_2D.getMaxLocation();
		final double[] min_pattern = pattern_2D.getMinLocation();
		final double range_pattern = max_pattern[0] - min_pattern[0];
		if (range_pattern == 0){
			return baseline_2D;
		}
		DenseDoubleMatrix2D scaled_2D = (DenseDoubleMatrix2D) pattern_2D.copy();
		scaled_2D.assign(contrast_2D, new DoubleDoubleFunction() {
			@Override
			public double apply(double pattern_val, double constrast_val) {
				double scaled_val =  (pattern_val - min_pattern[0]) / range_pattern;
				scaled_val = 2.0 * (scaled_val - 0.5);
				return (baseline_val * (1.0 + contrast_val * scaled_val));
			}
		});
		return scaled_2D;
	}

}
