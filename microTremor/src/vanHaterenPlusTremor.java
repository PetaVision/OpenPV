import java.awt.image.DataBuffer;
import java.util.Random;
import java.util.Vector;

import javax.media.jai.JAI;
import javax.media.jai.PlanarImage;
import javax.swing.JFrame;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.jet.math.tdouble.DoubleFunctions;
import flanagan.integration.RungeKutta;
import flanagan.plot.PlotGraph;

// class for managing input to van Hateren retina with ocular tremor added
@SuppressWarnings("restriction")
public class vanHaterenPlusTremor {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		// parse input arguments
		CommandLine commandLine;
		Options options = new Options();
		CommandLineParser parser = new GnuParser();
		String[] testArgs = { "--num_cones=32", "--image_type_id=0",
				"--grating_orientation=0", "--image_file=''", "--num_steps=64",
				"--delta_t=1.0", "--background_luminance=100.0",
				"--num_pixels=256", "--ran_seed=1234987" };

		OptionBuilder.withArgName("ran_seed");
		OptionBuilder.hasArg();
		OptionBuilder.withDescription("random number seed");
		Option option_ran_seed = OptionBuilder.create("ran_seed");
		options.addOption(option_ran_seed);

		OptionBuilder.withArgName("num_cones");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("number of cones in (square) retina along each side: rounded to power of 2");
		Option option_num_cones = OptionBuilder.create("num_cones");
		options.addOption(option_num_cones);
		int num_cones = 32;

		OptionBuilder.withArgName("num_pixels");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("number of pixels per cone, images should have higher resolution than cone array to describe micro tremor");
		Option option_num_pixels = OptionBuilder.create("num_pixels");
		options.addOption(option_num_pixels);
		int num_pixels = 8 * num_cones;

		int image_type_id = ImageType.ORIENTED_GRATING.ordinal();
		OptionBuilder.withArgName("image_type_id");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("image_type_id type [0=vertical_grating|1=disk_file|2=movie_file]");
		Option option_image_type_id = OptionBuilder.create("image_type_id");
		options.addOption(option_image_type_id);
		// ImageType image_type = ImageType.ORIENTED_GRATING;

		OptionBuilder.withArgName("grating_orientation");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("orientation of sinusoidal grating with Gaussian envelope");
		Option option_grating_orientation = OptionBuilder
				.create("grating_orientation");
		options.addOption(option_grating_orientation);
		double grating_orientation = Math.PI / 2;

		OptionBuilder.withArgName("image_file");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("pathname of image file or list of image frames");
		Option option_image_file = OptionBuilder.create("image_file");
		options.addOption(option_image_file);
		String image_file = "";

		OptionBuilder.withArgName("num_steps");
		OptionBuilder.hasArg();
		OptionBuilder
				.withDescription("number of time steps: rounded to power of 2");
		Option option_num_steps = OptionBuilder.create("num_steps");
		options.addOption(option_num_steps);
		int num_steps = 64;

		OptionBuilder.withArgName("delta_t");
		OptionBuilder.hasArg();
		OptionBuilder.withDescription("size of time step (msec)");
		Option option_delta_t = OptionBuilder.create("delta_t");
		options.addOption(option_delta_t);
		double delta_t = 1.0;

		OptionBuilder.withArgName("background_luminance");
		OptionBuilder.hasArg();
		OptionBuilder.withDescription("background luminance (Trolands)");
		Option option_background_luminance = OptionBuilder
				.create("background_luminance");
		options.addOption(option_background_luminance);
		double background_luminance = 100.0;

		Random generator = null;
		try {
			commandLine = parser.parse(options, testArgs); // args);

			if (commandLine.hasOption("ran_seed")) {
				System.out.print("ran_seed is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("ran_seed"));
				generator = new Random(Integer.parseInt(commandLine
						.getOptionValue("ran_seed")));
			} else {
				generator = new Random();
			}

			if (commandLine.hasOption("num_cones")) {
				System.out.print("num_cones is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("num_cones"));
				num_cones = Integer.parseInt(commandLine
						.getOptionValue("num_cones"));
				num_cones = (int) Math.pow(2,
						Math.floor(Math.log(num_cones) / Math.log(2)));
			}

			if (commandLine.hasOption("num_pixels")) {
				System.out.print("num_pixels is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("num_pixels"));
				num_pixels = Integer.parseInt(commandLine
						.getOptionValue("num_pixels"));
				num_pixels = (int) Math.pow(2,
						Math.floor(Math.log(num_pixels) / Math.log(2)));
			}

			if (commandLine.hasOption("image_type_id")) {
				System.out.print("image_type is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("image_type_id"));
				image_type_id = Integer.parseInt(commandLine
						.getOptionValue("image_type_id"));
			}

			if (image_type_id == ImageType.ORIENTED_GRATING.ordinal()) {
				if (commandLine.hasOption("grating_orientation")) {
					System.out
							.print("grating_orientation is present.  The value is: ");
					System.out.println(commandLine
							.getOptionValue("grating_orientation"));
					grating_orientation = Double.parseDouble(commandLine
							.getOptionValue("grating_orientation"));
				}
			}

			if (image_type_id == ImageType.DISK_FILE.ordinal()
					|| image_type_id == ImageType.MOVIE_FILE.ordinal()) {
				if (commandLine.hasOption("image_file")) {
					System.out.print("image_file is present.  The value is: ");
					System.out
							.println(commandLine.getOptionValue("image_file"));
					image_file = commandLine.getOptionValue("image_file");
				}
			}

			if (commandLine.hasOption("num_steps")) {
				System.out.print("num_steps is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("num_steps"));
				num_steps = Integer.parseInt(commandLine
						.getOptionValue("num_steps"));
				System.out.println("num_steps = " + Long.toString(num_steps));
				double log_num_steps = Math.log(num_steps);
				System.out.println("log_num_steps = "
						+ Double.toString(log_num_steps));
				double log2_num_steps = log_num_steps / Math.log(2);
				System.out.println("log2_num_steps = "
						+ Double.toString(log2_num_steps));
				num_steps = (int) Math.pow(2, Math.floor(log2_num_steps));
				System.out.println("num_steps = " + Long.toString(num_steps));
				// num_steps = (int) Math.pow(
				// Math.floor(Math.log(num_steps) / Math.log(2)), 2);
			}

			if (commandLine.hasOption("delta_t")) {
				System.out.print("delta_t is present.  The value is: ");
				System.out.println(commandLine.getOptionValue("delta_t"));
				delta_t = Float.parseFloat(commandLine
						.getOptionValue("delta_t"));
			}

			{
				String[] remainder = commandLine.getArgs();
				System.out.print("Remaining arguments: ");
				for (String argument : remainder) {
					System.out.print(argument);
					System.out.print(" ");
				}

				System.out.println();
			}

		} catch (ParseException exception) {
			System.out.print("Parse error: ");
			System.out.println(exception.getMessage()); // if no command line,
														// error message if no
														// default value
														// provided
		} // end parse input args

		// see: J. Hirsch and C. A. Curcio, The spatial resolution capacity of
		// human foveal retina, Vision Res. 29, 1989, 1095-I 101.
		// int num_cones = 32;
		double pixel_cone_ratio = num_pixels / num_cones;
		double pixel_diameter = RetinalConstants.coneDiameter
				/ pixel_cone_ratio;
		double minutes_per_cone = 60 * RetinalConstants.coneDiameter
				/ RetinalConstants.micronsPerDegree;
		double minutes_per_pixel = minutes_per_cone / pixel_cone_ratio;

		// random fluctuations
		// from An analysis of voltage noise in rod bipolar cells of the dogfish
		// retina. Ashmore JF, Falk G. J Physiol. 1982 Nov;332:273-97.
		// http://www.pubmedcentral.nih.gov/pagerender.fcgi?artid=1197398&pageindex=1#page
		// double random_contrast = 0.1 * background_luminance;

		// !!!TODO!!! what we need to do here is initialize a 3D matrix that
		// stores a movie, which could consist of a computer generated
		// pattern (i.e. gabor + 1/f noise) or a single image or an actual
		// movie. Jitter would then be added to this 3D matrix.
		PlanarImage input_planar_image = null;
		DenseDoubleMatrix2D input_matrix2D = null;
		if (image_type_id == ImageType.ORIENTED_GRATING.ordinal()) {
			// get 1/f background and gabor forgeround and then combine
			DenseDoubleMatrix2D bckgrnd_1_over_f = OneOverSpatialFreqBackgroundImage
					.getBackgroundImage(num_cones, num_pixels);
			DenseDoubleMatrix2D oriented_grating = GratingImage
					.getGratingImage(num_cones, num_pixels, Math.PI / 2, 20.0,
							1.0, 2.0, 0.0);
			input_matrix2D = (DenseDoubleMatrix2D) bckgrnd_1_over_f.copy();
			input_matrix2D.assign(oriented_grating, DoubleFunctions.plus);
			input_matrix2D.assign(MyUtils.convertDoubleToByte(input_matrix2D
					.elements()));
			input_planar_image = MyUtils.draw(input_matrix2D, num_pixels,
					num_pixels, "input image", null);
		} else if (image_type_id == ImageType.DISK_FILE.ordinal()) {
			input_planar_image = JAI.create("fileload", image_file)
					.createSnapshot();
			input_matrix2D = MyUtils.image2mat(input_planar_image);
			MyUtils.display(input_planar_image, image_file, null);
		} else if (image_type_id == ImageType.MOVIE_FILE.ordinal()) {
			// !!!TODO!!! Impliment this option
			input_planar_image = JAI.create("fileload", image_file)
					.createSnapshot();
			MyUtils.display(input_planar_image, image_file, null);
		}

		// init van Hatteren cone model
		vanHaterenCoupled vanHateren = new vanHaterenCoupled(num_cones,
				background_luminance);
		vanHateren.setHCKernel();
		DenseDoubleMatrix3D y_step3D = (DenseDoubleMatrix3D) vanHateren
				.getYInit3D().copy();

		// make tremor time series
		OcularTremor ocularTremor = new OcularTremor(generator);
		DenseDComplexMatrix1D tremor_time_series = ocularTremor.getTimeSeries(
				delta_t, num_steps);

		// init movie for storing jittered input
		JFrame tremor_frame = null;
		Vector<DenseDoubleMatrix2D> jittered_movie = new Vector<DenseDoubleMatrix2D>();
		jittered_movie.addElement(input_matrix2D);

		// get kernel for downsampling pixel image to discrete cone array
		DenseDoubleMatrix2D pixels_to_cones_kernel = MyUtils
				.getDownsampleKernel(num_pixels, num_cones);

		// init numerical integration engine
		RungeKutta runge_kutta = new RungeKutta();
		Vector<DenseDoubleMatrix3D> y_store = new Vector<DenseDoubleMatrix3D>();
		y_store.addElement(y_step3D);
		DenseDoubleMatrix2D jittered_input2D = (DenseDoubleMatrix2D) input_matrix2D.copy();
		DenseDoubleMatrix2D downsampled_input2D= MyUtils.downsample(
				jittered_input2D, pixels_to_cones_kernel);;
		for (int i_step = 1; i_step < num_steps; i_step++) {
			// add tremor to image
			jittered_input2D = OcularTremor
					.jitterImageUsingTremor(input_matrix2D, tremor_time_series,
							i_step, pixel_cone_ratio);
			downsampled_input2D = MyUtils.downsample(
					jittered_input2D, pixels_to_cones_kernel);
			tremor_frame = MyUtils.display(
					MyUtils.mat2image(downsampled_input2D), "tremor movie",
					tremor_frame);
			jittered_movie.addElement(jittered_input2D);
			vanHateren.setIexp(downsampled_input2D);
			runge_kutta.setInitialValueOfX((i_step - 1) * delta_t);
			runge_kutta.setFinalValueOfX(i_step * delta_t);
			runge_kutta.setInitialValuesOfY(y_step3D.elements());
			runge_kutta.setStepSize(delta_t);
			y_step3D.assign(runge_kutta.fourthOrder(vanHateren));
			y_store.addElement((DenseDoubleMatrix3D) y_step3D.copy());
		}
		
		// play jittered_input_movie
		
		// plot van Hatteren model responses
		int cone_row_index = num_cones / 2;
		int cone_col_index = num_cones / 2;
		double[] time_steps = new double[num_steps];
		for (int i_step = 0; i_step < time_steps.length; i_step++) {
			time_steps[i_step] = i_step;
		}
		DenseDoubleMatrix1D van_Hatteren_data = new DenseDoubleMatrix1D(num_steps);
		Vector<PlotGraph> van_Hatteren_plot = new Vector<PlotGraph>(y_step3D.slices());
		Vector<String> van_Hattern_titles = vanHaterenCoupled.getTitles(y_step3D.slices());

		for (int i_slice = 0; i_slice < y_step3D.slices(); i_slice++){
			van_Hatteren_data.assign(0.0);
			for (int i_step = 0; i_step < time_steps.length; i_step++) {
				van_Hatteren_data.set(i_step, y_store.get(i_step).get(i_slice, cone_row_index, cone_col_index));
			}
			van_Hatteren_plot.add(new PlotGraph(time_steps, van_Hatteren_data.toArray()));
			van_Hatteren_plot.get(i_slice).setLine(3);
			van_Hatteren_plot.get(i_slice).setGraphTitle(van_Hattern_titles.get(i_slice));
			van_Hatteren_plot.get(i_slice).setXaxisLegend("Time (msec)");		
			van_Hatteren_plot.get(i_slice).plot();
			van_Hatteren_plot.get(i_slice).setCloseChoice(2);
		}

	} // end main()
	
	

} // end class definition
