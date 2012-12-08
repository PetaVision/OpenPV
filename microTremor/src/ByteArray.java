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

import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import com.sun.media.jai.widget.DisplayJAI;

@SuppressWarnings("restriction")
public class ByteArray {

	@SuppressWarnings("restriction")
	public static PlanarImage draw(DenseDoubleMatrix2D matrix_2D, int num_rows,
			int num_cols, String title_string, JFrame frame) {

		PlanarImage image_snap = ByteArray.mat2image(matrix_2D);
		ByteArray.display(image_snap, title_string, frame);
		return image_snap;
	}

	public static JFrame display(PlanarImage image_snap, String title_string, JFrame frame) {
		if (frame == null){
			frame = new JFrame();
		}
		frame.setTitle(title_string);
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setSize(image_snap.getHeight() / 2, image_snap.getWidth() / 2);
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
		matrix_2D.assign(ByteArray.convertDoubleToByte(matrix_2D
				.elements()));
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

}
