import java.util.Vector;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.jet.math.tdouble.DoubleFunctions;
import flanagan.integration.DerivnFunction;

public class vanHaterenCoupled implements DerivnFunction {

	public DoubleMatrix2D Iexp; // should set Iexp at beginning and at end of
								// interval and use interpolation to find value
								// at time t in between

	int numCones; // number of cones in each dimension
	int numStateVars = 10;

	// Outer segment: Phototransduction cascade
	double tau_R = 3.4; // ms, range 0.5 - 6.5
	double A_R = 0.1;
	double tau_E = 8.7; // ms, range 3.0 - 16.8
	double A_E = 0.3395;
	double c_beta = 0.0028;  // (ms)^-1, range 2.0 - 4.0 * 10^-3
	double k_beta = 0.00016; // (ms)^-1, range 4.9 - 39.0 * 10^-5

	// Outer segment: Calcium feedback
	double eta = 0.38;
	double n_X = 1.0;
	double tau_C = 3.0;
	double a_C = 0.09;
	double n_C = 4.0;

	// Inner segment
	double tau_m = 4.0;
	double a_is = 0.07;
	double gamma = 0.7;
	double tau_is = 90.0;

	// Horizontal cell feedback
	double tau_a = 250.0;
	double V_I = 20.0;
	double mu = 0.7;
	double g_t = 125.0;
	double V_k = -10.0;
	double V_n = 3.0;
	double tau_1 = 4.0;
	double tau_2 = 4.0;
	double tau_h = 20.0;

	// kernel function of horizontal cell interactions
	double V_h_coupling_const = 1; // roughly: ratio of coupling conductance to
									// leak conductance
	double hc_lamda = 4; // sigma of HC spatial Gaussian (in cone units)
	DoubleMatrix2D hcKernel; //
	double[] yInit;
	DenseDoubleMatrix3D yInit3D;
	double backgroundLuminance;

	static double[] y_init_default = { 34.0000, // R
			100.4242, // E
			12.9313, // X
			14.7417, // C
			21.5482, // V_is
			0.6000, // g_i
			21.5200, // V_is_2
			34.2400, // V_1
			34.2460, // V_b
			34.2720 // V_h
	};
	static double[] y_init_100 = { 34.0000, 100.4241, 12.9313, 14.7417,
			21.5408, 0.6003, 21.5408, 34.2475, 34.2475, 34.2475 };
	static double[] y_init_10 = { 3.4000, 10.0424, 17.9921, 20.5110, 26.1600,
			0.6878, 26.1585, 37.8198, 37.8198, 37.8198, };
	static double[] y_init_1 = { 0.3400, 1.0042, 19.5713, 22.3113, 27.4872,
			0.7120, 27.4851, 38.8509, 38.8509, 38.8509 };

	public vanHaterenCoupled(int num_cones, double background_luminance) {
		super();
		numCones = num_cones;
		setBackgroundLuminance(background_luminance);
		// set initial conditions
		// initial values used by Furusawa and Kamiyama
		if (background_luminance == 100.0) {
			yInit = y_init_100;
		} else if (background_luminance == 10.0) {
			yInit = y_init_10;
		} else if (background_luminance == 1.0) {
			yInit = y_init_1;
		} else {
			yInit = y_init_default;
		}
		numStateVars = yInit.length;
		yInit3D = new DenseDoubleMatrix3D(numStateVars, num_cones, num_cones);
		for (int i_var = 0; i_var < numStateVars; i_var++) {
			yInit3D.viewSlice(i_var).assign(yInit[i_var]);
		}
	}

	public static double[] getYInit_default() {
		return y_init_default;
	}

	public static double[] getYInit_100() {
		return y_init_100;
	}

	public static double[] getYInit_10() {
		return y_init_10;
	}

	public static double[] getYInit_1() {
		return y_init_1;
	}

	public DenseDoubleMatrix3D getYInit3D() {
		return yInit3D;
	}

	public static Vector<String> getTitles(int num_slices) {
		Vector<String> van_Hattern_titles = new Vector<String>();
		van_Hattern_titles.add("R");
		van_Hattern_titles.add("E");
		van_Hattern_titles.add("X");
		van_Hattern_titles.add("C");
		van_Hattern_titles.add("V_is");
		van_Hattern_titles.add("g_i");
		van_Hattern_titles.add("V_is_2");
		van_Hattern_titles.add("V_1");
		van_Hattern_titles.add("V_b");
		van_Hattern_titles.add("V_h");
		return van_Hattern_titles;
	}

	@Override
	public double[] derivn(double t, double[] y_init) {
		double[] dydt = new double[y_init.length]; // output format
		DenseDoubleMatrix3D y_init3D = new DenseDoubleMatrix3D(numStateVars, numCones,
				numCones);
		y_init3D.assign(y_init); // assume y_init is produced by
									// y_init3D.toArray()

		// state variables
		DenseDoubleMatrix2D R = new DenseDoubleMatrix2D(y_init3D.viewSlice(0)
				.toArray());
		DenseDoubleMatrix2D E = new DenseDoubleMatrix2D(y_init3D.viewSlice(1)
				.toArray());
		DenseDoubleMatrix2D X = new DenseDoubleMatrix2D(y_init3D.viewSlice(2)
				.toArray());
		DenseDoubleMatrix2D C = new DenseDoubleMatrix2D(y_init3D.viewSlice(3)
				.toArray());
		DenseDoubleMatrix2D V_is = new DenseDoubleMatrix2D(y_init3D.viewSlice(4)
				.toArray());
		DenseDoubleMatrix2D g_i = new DenseDoubleMatrix2D(y_init3D.viewSlice(5)
				.toArray());
		DenseDoubleMatrix2D V_is_2 = new DenseDoubleMatrix2D(y_init3D.viewSlice(6)
				.toArray());
		DenseDoubleMatrix2D V_1 = new DenseDoubleMatrix2D(y_init3D.viewSlice(7)
				.toArray());
		DenseDoubleMatrix2D V_b = new DenseDoubleMatrix2D(y_init3D.viewSlice(8)
				.toArray());
		DenseDoubleMatrix2D V_h = new DenseDoubleMatrix2D(y_init3D.viewSlice(9)
				.toArray());

		DenseDoubleMatrix2D alpha = (DenseDoubleMatrix2D) C.copy().assign(new DoubleFunction() {

			@Override
			public double apply(double C_val) {
				return (1.0 / (1.0 + Math.pow((a_C * C_val), n_C)));
			}
		});

		// DoubleMatrix2D beta = c_beta + k_beta*E;
		DenseDoubleMatrix2D beta = (DenseDoubleMatrix2D) E.copy().assign(new DoubleFunction() {

			@Override
			public double apply(double E_val) {
				return (c_beta + k_beta * E_val);
			}
		});

		// DoubleMatrix2D tau_X = 1 ./ beta;
		DenseDoubleMatrix2D tau_X = (DenseDoubleMatrix2D) beta.copy().assign(DoubleFunctions.inv);

		// DoubleMatrix2D I_os = X.^n_X;
		DenseDoubleMatrix2D I_os = (DenseDoubleMatrix2D) X.copy().assign(DoubleFunctions.pow(n_X));

		// DoubleMatrix2D g_is = a_is * ( V_is.^gamma );
		DenseDoubleMatrix2D g_is = (DenseDoubleMatrix2D) V_is.copy().assign(new DoubleFunction() {

			@Override
			public double apply(double V_is_val) {
				return (a_is * Math.pow(V_is_val, gamma));
			}
		});

		// DoubleMatrix2D a_I = ( V_is_2 ./ V_I).^mu;
		DenseDoubleMatrix2D a_I = (DenseDoubleMatrix2D) V_is_2.copy().assign(new DoubleFunction() {

			@Override
			public double apply(double V_is_2_elelment) {
				return Math.pow((V_is_2_elelment / V_I), mu);
			}
		});

		// DoubleMatrix2D V_s = V_is - V_h;
		DenseDoubleMatrix2D V_s = (DenseDoubleMatrix2D) V_is.copy().assign(V_h, DoubleFunctions.minus);

		// DoubleMatrix2D I_t = ( g_t ./ ( a_I .* ( 1 + exp( -( V_s - V_k ) /
		// V_n) ) ) );
		DenseDoubleMatrix2D I_t = (DenseDoubleMatrix2D) V_s.copy().assign(a_I, new DoubleDoubleFunction() {

			@Override
			public double apply(double V_s_val, double a_I_val) {
				return (g_t / (a_I_val * (1 + Math.exp(-(V_s_val - V_k) / V_n))));
			}
		});

		// ydot
		DenseDoubleMatrix3D ydot_3D = new DenseDoubleMatrix3D(numStateVars, numCones,
				numCones);

		// ydot_R = A_R * Iexp - R / tau_R;
		DenseDoubleMatrix2D ydot_R = (DenseDoubleMatrix2D) R.copy();
		ydot_R.assign(Iexp, new DoubleDoubleFunction() {

			@Override
			public double apply(double R_val, double Iexp_val) {
				return (A_R * Iexp_val - R_val / tau_R);
			}
		});
		ydot_3D.viewSlice(0).assign(ydot_R);

		// ydot_E = A_E * R - E / tau_E;
		DenseDoubleMatrix2D ydot_E = (DenseDoubleMatrix2D) E.copy().assign(R, new DoubleDoubleFunction() {

			@Override
			public double apply(double E_val, double R_val) {
				return (A_E * R_val - E_val / tau_E);
			}
		});
		ydot_3D.viewSlice(1).assign(ydot_E);

		// ydot_X = ( alpha ./ beta - X ) ./ tau_X;
		DenseDoubleMatrix2D alpha_over_beta = (DenseDoubleMatrix2D) alpha.copy().assign(beta,
				DoubleFunctions.div);
		DenseDoubleMatrix2D alpha_over_beta_minus_X = (DenseDoubleMatrix2D) alpha_over_beta.copy().assign(
				X, DoubleFunctions.minus);
		DenseDoubleMatrix2D ydot_X = (DenseDoubleMatrix2D) alpha_over_beta_minus_X.copy().assign(tau_X,
				DoubleFunctions.div);
		ydot_3D.viewSlice(2).assign(ydot_X);

		// ydot_C = eta * I_os - C / tau_C;
		DenseDoubleMatrix2D ydot_C = (DenseDoubleMatrix2D) C.copy().assign(I_os,
				new DoubleDoubleFunction() {

					@Override
					public double apply(double C_val, double I_os_val) {
						return (eta * I_os_val - C_val / tau_C);
					}
				});
		ydot_3D.viewSlice(3).assign(ydot_C);

		// ydot_V_is = ( I_os ./ g_i - V_is ) / tau_m;
		DenseDoubleMatrix2D I_os_over_g_i = (DenseDoubleMatrix2D) I_os.copy().assign(g_i,
				DoubleFunctions.div);
		DenseDoubleMatrix2D ydot_V_is = (DenseDoubleMatrix2D) V_is.copy().assign(I_os_over_g_i,
				new DoubleDoubleFunction() {

					@Override
					public double apply(double V_is_val,
							double I_os_over_g_i_val) {
						return ((I_os_over_g_i_val - V_is_val) / tau_m);
					}
				});
		ydot_3D.viewSlice(4).assign(ydot_V_is);

		// ydot_g_i = ( g_is - g_i ) / tau_is;
		DenseDoubleMatrix2D ydot_g_i = (DenseDoubleMatrix2D) g_i.copy().assign(g_is,
				new DoubleDoubleFunction() {

					@Override
					public double apply(double g_i_val, double g_is_val) {
						return ((g_is_val - g_i_val) / tau_is);
					}
				});
		ydot_3D.viewSlice(5).assign(ydot_g_i);

		// ydot_V_is_2 = ( V_is - V_is_2 ) / tau_a;
		DenseDoubleMatrix2D ydot_V_is_2 = (DenseDoubleMatrix2D) V_is_2.copy().assign(V_is,
				new DoubleDoubleFunction() {

					@Override
					public double apply(double V_is_2_val, double V_is_val) {
						return ((V_is_val - V_is_2_val) / tau_a);
					}
				});
		ydot_3D.viewSlice(6).assign(ydot_V_is_2);

		// ydot_V_1 = ( I_t - V_1 ) / tau_1;
		DoubleMatrix2D ydot_V_1 = V_1.copy().assign(I_t,
				new DoubleDoubleFunction() {

					@Override
					public double apply(double V_1_val, double I_t_val) {
						return ((I_t_val - V_1_val) / tau_1);
					}
				});
		ydot_3D.viewSlice(7).assign(ydot_V_1);

		// ydot_V_b = ( V_1 - V_b ) ./ (a_I*tau_2);
		DoubleMatrix2D ydot_V_b = V_1.assign(V_b, DoubleFunctions.minus).assign(a_I, 
				new DoubleDoubleFunction() {

					@Override
					public double apply(double V_diff_val, double a_I_val) {
						return (V_diff_val / (a_I_val*tau_2));
					}
				});
		ydot_3D.viewSlice(8).assign(ydot_V_b);

		// convolution of V_h
		// V_h_2D = reshape( V_h, [num_cones, num_cones] );
		// V_h_2D = hc_kernel * V_h_2D * hc_kernel;
		DenseDoubleAlgebra dbl_algebra = new DenseDoubleAlgebra();
		DenseDoubleMatrix2D V_h_gap2D = (DenseDoubleMatrix2D) dbl_algebra.mult(hcKernel,
					dbl_algebra.mult(V_h, hcKernel.viewDice().copy())).copy();
//		double V_h_gap_mean = V_h_gap2D.zSum() / (numCones * numCones);
		// V_h2 = reshape( V_h_2D, [num_cones2, 1] );
		// V_h2 = mean( V_h );

		// added HC coupling so that if V_h2 == V_h (i.e. for a uniform input),
		// then should get identical behavior to the original van Hatteren model
		// implicitly assumes that tau_h has been renormalized so that it
		// remains invariant wrt V_h_coupling_const
		
		// ydot_V_h = ...
		// ( V_b + V_h_coupling_const * ( V_h2 - V_h ) - V_h ) ./ ...
		// (a_I*tau_h);
		DoubleMatrix2D ydot_V_h = V_h.assign(V_h_gap2D, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double V_h_val, double V_h_gap2D_val) {
				return (V_h_coupling_const * ( V_h_gap2D_val - V_h_val ) - V_h_val);
			}
		});
		ydot_V_h = ydot_V_h.assign(V_b, DoubleFunctions.plus);
		ydot_3D.viewSlice(9).assign(ydot_V_h);

		dydt = ydot_3D.elements();
		return dydt;
	}

	public void setIexp(DoubleMatrix2D I_exp) {
		this.Iexp = I_exp;
	}

	// assume Gaussian fall off of gap junction strength
	// should actually be sum of two Gaussians to better match physiological data
	public void setHCKernel() {
		DenseDoubleMatrix2D hc_kernel = new DenseDoubleMatrix2D(numCones, numCones);
		double[][] kernel_vals = new double[numCones][numCones];
		for (int i_pre = 0; i_pre < numCones; i_pre++) {
			for (int i_post = 0; i_post < numCones; i_post++) {
				int dist_abs = Math.abs(i_post - i_pre);
				dist_abs = (dist_abs < numCones/2) ? dist_abs : numCones - dist_abs;
				kernel_vals[i_pre][i_post] = (Math.exp(-Math.pow(dist_abs, 2)
						/ (2 * Math.pow(hc_lamda, 2))));
			}
		}
		hc_kernel.assign(kernel_vals);
		double hc_kernel_norm = hc_kernel.zSum() / numCones; // normalizes total gap junction input to one
		hc_kernel.assign(hc_kernel.copy().assign(hc_kernel_norm), DoubleFunctions.div);
		hcKernel = hc_kernel;
	}

	public void setBackgroundLuminance(double background_luminance) {
		backgroundLuminance = background_luminance;
	}

	public DoubleMatrix2D getHCKernel() {
		return hcKernel;
	}

}
