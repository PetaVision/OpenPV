%=Neural tuning parameters
%
NX=36 % 128
NY=36 % 128
NO=8
NK=4
DTH=(180.0/NO)

MIN_DX=1.e-8
DX=1.0		
DY=1.0		
DTH=(180.0/NO)	
K_0=0.0                   %  curvature start at zero (straight line)
DK=(1.0/(6*(NK-1)))            %  change in curvature
SIG_C_K_x2=2.0*DK*DK        % tolerance from target curvature:l->kappa
MIN_DENOM=1.0E-10               % 1.0E-10;
MS_PER_TIMESTEP=1.0





POISSON_INPUT=0			% 0=off: continuous spikes. 1=on, Poisson
POISSON_INPUT_RATE=1000.0	 		% Firing rate of "on" edges in Hz
POISSON_RATE_TIMESTEPS=((POISSON_INPUT_RATE*MS_PER_TIMESTEP)/1000.0)





%Excitation=basics
V_TH_0=0.5			  % threshold potential
DT_d_TAU=0.125		          % rate of change of excitation
ALPHA=0.01		          % desired fraction active per time step
NOISE_AMP=(1.0*0.5*V_TH_0/DT_d_TAU) % maximum amplitude of noise if present
NOISE_FREQ=0.5  %0.5                % prob of noise input on each time step
MIN_V=-4.0*V_TH_0               %minimum potential


%Inhibition=basics
INHIBIT_ON=1%inhibition flag, define to turn on inhibition
DT_d_TAU_INH=0.065  % (DT_d_TAU/2)    % rate of change of inhibition
V_TH_0_INH=1.0*V_TH_0              %  threshold potential for inhibitory neurons
NOISE_AMP_INH=NOISE_AMP               % maximum amplitude of noise if present
NOISE_FREQ_INH=NOISE_FREQ              % prob of noise input on each time step
MIN_H=-1.0*V_TH_0_INH             %minimum inhibitory potential








INHIB_FRACTION=0.9                                               % fraction of inhibitory connections
WEIGHT_SCALE=0.033*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))

%Excite=to Excite connection
SIG_C_D_x2=(2*4.0*4.0)	                                   % (squared and times 2)
SIG_C_P_x2=(2*1.0*DTH*DTH)
COCIRC_SCALE=(1.0*WEIGHT_SCALE)                                     % Scale for Excite to excite cells
%COCIRC_SCALE=(0.5*V_TH_0/DT_d_TAU)	                           % (.025,0)->stable, (.05,0)->unstable
EXCITE_R2=8*8*(DX*DX+DY*DY)                                  % cut-off radius for excititory cells(infinite wrt the screen size for now)
INHIBIT_SCALE=0*1.0	                                           % reduce inhibition (w < 0) by this amount


%Inhibit=to Excite connection
INHIB_DELAY=3                      %number of times steps delay (x times slower than excititory conections)
SIG_I_D_x2=(2*2.0*2.0)            % sigma (square and time 2) for inhibition to exicititory connections
SIG_I_P_x2=(2*1.0*DTH*DTH)
INHIB_R2=4.0*4.0*(DX*DX+DY*DY)  %square of radius of inhibition
SCALE_INH=(-125.0*WEIGHT_SCALE)
INHIB_FRACTION_I=0.8                    % fraction of inhibitory connections
INHIBIT_SCALE_I=0*1.0	                 % reduce inhibition (w < 0) by this amount


%Inhibition=of the inhibition
SIG_II_D_x2=SIG_I_D_x2          %sigma (square and time 2) for inhibition to exicititory connections
SIG_II_P_x2=SIG_I_P_x2
INHIBI_R2=INHIB_R2            %square of radius of inhibition
SCALE_INHI=(-10.0*WEIGHT_SCALE)
INHIB_FRACTION_II=0.8                    % fraction of inhibitory connections
INHIBIT_SCALE_II=0*1.0	                 % reduce inhibition (w < 0) by this amount


%Gap=Junctions
SIG_G_D_x2=(2*2.0*2.0)           %sigma (square and times 2) for gap junctions (inhibit to inhibit)
SIG_G_P_x2=(2*1.0*DTH*DTH)
GAP_R2=4.0*4.0*(DX*DX+DY*DY)         %square of radius of gap junctions keep small
SCALE_GAP=2.0*WEIGHT_SCALE
INHIB_FRACTION_G=0.9                   % fraction of inhibitory connections
INHIBIT_SCALE_G=0*1.0	                % reduce inhibition (w < 0) by this amount


%Excite=to Inhibit connection
SIG_E2I_D_x2=SIG_C_D_x2
SIG_E2I_P_x2=SIG_C_P_x2
E2I_R2=EXCITE_R2
E_TO_I_SCALE=6.0*WEIGHT_SCALE
INHIB_FRACTION_E2I=0.9             % fraction of onhibitory connections
INHIBIT_SCALE_E2I=0*1.0	    % reduce inhibition (w < 0) by this amount


%=Others:
I_MAX=1.0*(1.0*0.5*V_TH_0/DT_d_TAU) % maximum image intensity
CLUTTER_PROB=0.01            % prob of clutter in image

#endif=
