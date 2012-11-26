import java.awt.image.Kernel;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.jet.math.Functions;
import flanagan.integration.DerivnFunction;


public class vanHaterenCoupled implements DerivnFunction {

public
	double[][] photo_cone; // input image
	DoubleMatrix2D Iexp;
	int num_cones; // number of cones in each dimension
	int num_cones2 = num_cones*num_cones;
	int num_state_vars = 10;

	// Outer segment: Phototransduction cascade
	double tau_R   = 3.4;
	double A_R     = 0.1;
	double tau_E   = 8.7;
	double A_E     = 0.3395;
	double c_beta  = 0.0028;
	double k_beta  = 0.00016;

	// Outer segment: Calcium feedback
	double eta     = 0.38;
	double n_X     = 1.0;
	double tau_C   = 3.0;
	double a_C     = 0.09;
	double n_C     = 4.0;

	// Inner segment
	double tau_m   = 4.0;
	double a_is    = 0.07;
	double gamma   = 0.7;
	double tau_is  = 90.0;

	// Horizontal cell feedback
	double tau_a   = 250.0;
	double V_I     = 20.0;
	double mu      = 0.7;
	double g_t      = 125.0;
	double V_k     = -10.0;
	double V_n     = 3.0;
	double tau_1   = 4.0;
	double tau_2   = 4.0;
	double tau_h   = 20.0;
	
	 // kernel function of horizontal cell interactions
	double V_h_coupling_const = 1;  // roughly: ratio of coupling conductance to leak conductance
	double hc_lamda = 4;  //sigma of HC spatial Gaussian (in cone units)
	DoubleMatrix2D hcKernel; // 
	double[] yInit;	
	DenseDoubleMatrix3D yInit3D;	
	
	static double[] y_init_default  = {
			34.0000,  // R
			100.4242, // E
			12.9313,  // X
			14.7417,  // C
			21.5482,  // V_is
			0.6000,   // g_i
			21.5200,  // V_is_2
			34.2400,  // V_1
			34.2460,  // V_b
			34.2720 // V_h
	};
	static double[] y_init_100  = {
			34.0000,
			100.4241,
			12.9313,
			14.7417,
			21.5408,
			0.6003,
			21.5408,
			34.2475,
			34.2475,
			34.2475
	};
	static double[] y_init_10  = {
			3.4000,
			10.0424,
			17.9921,
			20.5110,
			26.1600,
			0.6878,
			26.1585,
			37.8198,
			37.8198,
			37.8198,
	};
	static double[] y_init_1  = {
			0.3400,
			1.0042,
			19.5713,
			22.3113,
			27.4872,
			0.7120,
			27.4851,
			38.8509,
			38.8509,
			38.8509
	};

	public vanHaterenCoupled() {
		super();
		double background_luminance = RetinalConstants.backgroundLuminance;
		// set initial conditions
		// initial values used by Furusawa and Kamiyama
		if (background_luminance == 100.0){
			yInit = y_init_100;
		}
		else if (background_luminance == 10.0){
			yInit = y_init_10;
		}
		else if (background_luminance == 1.0){
			yInit = y_init_1;
		}
		else {
			yInit = y_init_default;
		}
		yInit3D = new DenseDoubleMatrix3D(yInit.length, num_cones, num_cones);
		for (int i_var = 0; i_var < yInit.length; i_var++){
			yInit3D.viewSlice(i_var).assign(yInit[i_var]);
		}
	}
	
	public static double[] getYInit_default(){
		return y_init_default;
	}

	public static double[] getYInit_100(){
		return y_init_100;
	}

	public static double[] getYInit_10(){
		return y_init_10;
	}

	public static double[] getYInit_1(){
		return y_init_1;
	}
	
	public DenseDoubleMatrix3D getYInit3D(){
		return yInit3D;
	}

	@Override
	public double[] derivn(double t, double[] y_init) {
		double[] dydt = new double[y_init.length];  // output format
		DenseDoubleMatrix3D y_init3D = new DenseDoubleMatrix3D(num_cones, num_cones, num_state_vars);
		y_init3D.assign(y_init);  // assume y_init is produced by y_init3D.toArray()
		
		// state variables
		DoubleMatrix2D R       = new DenseDoubleMatrix2D(y_init3D.viewSlice(0).toArray());
		DoubleMatrix2D E       = new DenseDoubleMatrix2D(y_init3D.viewSlice(1).toArray());
		DoubleMatrix2D X       = new DenseDoubleMatrix2D(y_init3D.viewSlice(2).toArray());
		DoubleMatrix2D C       = new DenseDoubleMatrix2D(y_init3D.viewSlice(3).toArray());
		DoubleMatrix2D V_is    = new DenseDoubleMatrix2D(y_init3D.viewSlice(4).toArray());
		DoubleMatrix2D g_i     = new DenseDoubleMatrix2D(y_init3D.viewSlice(5).toArray());
		DoubleMatrix2D V_is_2  = new DenseDoubleMatrix2D(y_init3D.viewSlice(6).toArray());
		DoubleMatrix2D V_1     = new DenseDoubleMatrix2D(y_init3D.viewSlice(7).toArray());
		DoubleMatrix2D V_b     = new DenseDoubleMatrix2D(y_init3D.viewSlice(8).toArray());
		DoubleMatrix2D V_h     = new DenseDoubleMatrix2D(y_init3D.viewSlice(9).toArray());

		

		DoubleMatrix2D alpha = C.copy().assign(new DoubleFunction() {
			
			@Override
			public double apply(double C_val) {
				return (1.0 / ( 1.0 + Math.pow((a_C*C_val), n_C) ));
			}
		});
		
		//DoubleMatrix2D beta    = c_beta + k_beta*E;
		DoubleMatrix2D beta    = E.copy().assign(new DoubleFunction() {
			
			@Override
			public double apply(double E_val) {
				return (c_beta + k_beta*E_val);
			}
		});
		
		//DoubleMatrix2D tau_X   = 1 ./ beta;
		DoubleMatrix2D tau_X   = beta.copy().assign((DoubleFunction) Functions.inv);
		
		//DoubleMatrix2D I_os    = X.^n_X;
		DoubleMatrix2D I_os    = X.copy().assign((DoubleFunction) Functions.pow(n_X));
		
		//DoubleMatrix2D g_is    = a_is * ( V_is.^gamma );
		DoubleMatrix2D g_is    = V_is.copy().assign(new DoubleFunction() {
			
			@Override
			public double apply(double V_is_val) {
				return (a_is * Math.pow(V_is_val, gamma ));
			}
		});
		
		//DoubleMatrix2D a_I     = ( V_is_2 ./ V_I).^mu;
		DoubleMatrix2D a_I     = V_is_2.copy().assign(new DoubleFunction() {
			
			@Override
			public double apply(double V_is_2_elelment) {
				return Math.pow( (V_is_2_elelment / V_I), mu);
			}
		});
		
		//DoubleMatrix2D V_s     = V_is - V_h;
		DoubleMatrix2D V_s     = V_is.copy().assign(V_h, (DoubleDoubleFunction) Functions.minus);
		
		//DoubleMatrix2D I_t     = ( g_t ./ ( a_I .* ( 1 + exp( -( V_s - V_k ) / V_n) ) ) );
		DoubleMatrix2D I_t     = V_s.copy().assign(a_I, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double V_s_val, double a_I_val) {
				return ( g_t / ( a_I_val * ( 1 + Math.exp(-( V_s_val - V_k ) / V_n) ) ) );
			}
		});



		// convolution of V_h
		//V_h_2D = reshape( V_h, [num_cones, num_cones] );
		//V_h_2D = hc_kernel * V_h_2D * hc_kernel; 
		//V_h2 = reshape( V_h_2D, [num_cones2, 1] );
		// V_h2 = mean( V_h );


		// ydot
		DenseDoubleMatrix3D ydot_3D = new DenseDoubleMatrix3D(num_cones, num_cones, num_state_vars);

		// ydot_R = A_R * Iexp -  R / tau_R;
		DoubleMatrix2D ydot_R = R.copy().assign(Iexp, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double R_val, double Iexp_val) {
				return (A_R * Iexp_val -  R_val / tau_R);
			}
		}); 
		ydot_3D.viewSlice(0).assign(ydot_R);

		//ydot_E = A_E * R - E / tau_E;
		DoubleMatrix2D ydot_E = E.copy().assign(R, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double E_val, double R_val) {
				return (A_E * R_val - E_val / tau_E);
			}
		}); 
		ydot_3D.viewSlice(1).assign(ydot_E);
		
		
		//ydot_X = ( alpha ./ beta - X ) ./ tau_X;
		DoubleMatrix2D alpha_over_beta    = alpha.copy().assign(beta, (DoubleDoubleFunction) Functions.div);		
		DoubleMatrix2D alpha_over_beta_minus_X    = alpha_over_beta.copy().assign(X, (DoubleDoubleFunction) Functions.minus);		
		DoubleMatrix2D ydot_X = alpha_over_beta_minus_X.copy().assign(tau_X, (DoubleDoubleFunction) Functions.div); 
		ydot_3D.viewSlice(2).assign(ydot_X);
		
		//ydot_C = eta * I_os - C / tau_C;
		DoubleMatrix2D ydot_C = C.copy().assign(I_os, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double C_val, double I_os_val) {
				return (eta * I_os_val - C_val / tau_C);
			}
		}); 
		ydot_3D.viewSlice(3).assign(ydot_C);
		
		//ydot_V_is = ( I_os ./ g_i - V_is ) / tau_m;
		DoubleMatrix2D I_os_over_g_i    = I_os.copy().assign(g_i, (DoubleDoubleFunction) Functions.div);
		DoubleMatrix2D ydot_V_is = V_is.copy().assign(I_os_over_g_i, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double V_is_val, double I_os_over_g_i_val) {
				return (( I_os_over_g_i_val - V_is_val ) / tau_m);
			}
		}); 
		ydot_3D.viewSlice(4).assign(ydot_V_is);
		
		//ydot_g_i = ( g_is - g_i ) / tau_is;
		DoubleMatrix2D ydot_g_i = g_i.copy().assign(g_is, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double g_i_val, double g_is_val) {
				return (( g_is_val - g_i_val ) / tau_is);
			}
		}); 
		ydot_3D.viewSlice(5).assign(ydot_g_i);
		
		//ydot_V_is_2 = ( V_is - V_is_2 ) / tau_a;
		DoubleMatrix2D ydot_V_is_2 = V_is_2.copy().assign(V_is, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double V_is_2_val, double V_is_val) {
				return (( V_is_val - V_is_2_val ) / tau_a);
			}
		}); 
		ydot_3D.viewSlice(6).assign(ydot_V_is_2);
		
		//ydot_V_1 = ( I_t - V_1 ) / tau_1;
		DoubleMatrix2D ydot_V_1 = V_1.copy().assign(I_t, new DoubleDoubleFunction() {
			
			@Override
			public double apply(double V_1_val, double I_t_val) {
				return (( I_t_val - V_1_val ) / tau_1);
			}
		}); 
		ydot_3D.viewSlice(7).assign(ydot_V_1);
		
		//ydot_V_b = ( V_1 - V_b ) ./ (a_I*tau_2);
		//		ydot_V_h = ...
		//		    ( V_b + V_h_coupling_const * ( V_h2 - V_h ) - V_h ) ./ ...
		//		    (a_I*tau_h);
		//ydot_3D.viewSlice(8).assign(ydot_V_b);
		
		
		// added HC coupling so that if V_h2 == V_h (i.e. for a uniform input),
		// then should get identical behavior to the original van Hatteren model
		// implicitly assumes that tau_h has been renormalized so that it remains invariant wrt
		// V_h_coupling_const
		
		dydt = ydot_3D.elements();
		return dydt;
	}

	public void setIexp(DoubleMatrix2D I_exp) {
		this.Iexp = I_exp;
	}
	
	public void setHCKernel() {
		DoubleMatrix2D hc_kernel = new DenseDoubleMatrix2D(num_cones, num_cones);
		double[][] row_dist = new double[num_cones][num_cones];
		for (int i_pre = 0; i_pre < num_cones; i_pre++) {
			for (int i_post = 0; i_post < num_cones; i_post++) {
				row_dist[i_pre][i_post] = i_post - i_pre;
			}
		}
		
		hc_kernel.assign(row_dist);
		hc_kernel.assign(new DoubleFunction() {

			@Override
			public double apply(double row_dist) {
				if (row_dist > 0) {
					return (Math.exp(-Math.pow(row_dist, 2)
							/ (2 * Math.pow(hc_lamda, 2))));
				} else {
					return 0.0;
				}
			}
		});
		double hc_kernel_norm = hc_kernel.aggregate(new DoubleDoubleFunction() {
			
			@Override
			public double apply(double cum_sum, double kernel_val) {
				return (cum_sum + kernel_val);
			}
		}, new DoubleFunction() {
			
			@Override
			public double apply(double kernel_val) {
				return kernel_val;
			}
		});		
//		double hc_kernel_norm = hc_kernel.aggregate((DoubleDoubleFunction) Functions.plus, (DoubleFunction)Functions.identity);
//		hc_kernel_norm = Math.pow(hc_kernel_norm, 2);
//		hc_kernel_norm = Math.sqrt(hc_kernel_norm);
	
		hc_kernel.assign(hc_kernel.copy().assign(hc_kernel_norm), new DoubleDoubleFunction() {
			
			@Override
			public double apply(double hc_kernel_val, double norm_val) {
				return norm_val > 0 ? (hc_kernel_val / norm_val) : hc_kernel_val;
			}
		});
//		hc_kernel.assign((DoubleFunction) Functions.div(hc_kernel_norm));
		hcKernel = hc_kernel;
	}
	
	public void setBackgroundLuminance(){
		
	}
	
	public DoubleMatrix2D getHCKernel(){
		return hcKernel;
	}
}
