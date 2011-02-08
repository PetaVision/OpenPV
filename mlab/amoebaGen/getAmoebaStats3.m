function getAmoebaStats3(trial_ndx, nfourier)

  global image_dim
  global mean_amoeba_xy std_amoeba_xy mean_distractor_xy std_distractor_xy
  global trial num_trials
  trial = trial_ndx;

amoeba_struct = struct;
%rand('twister', sum(100*clock));  % rand automatically initialize "randomly"
%amoeba_struct.rand_state = {rand('twister')};
amoeba_struct.rand_state = {rand('state')};
amoeba_struct.num_segments = 2^4; % 2^3; % 
amoeba_struct.image_rect_size = image_dim(1);

amoeba_struct.num_targets = 1;
%amoeba_struct.num_distractors = 0*amoeba_struct.num_targets; %for targets
amoeba_struct.num_distractors = 2; %with amoeba targets
amoeba_struct.segments_per_distractor =  2^(-3); % 2^(-2); % %as fraction
				% of num_segments, 2nd value used in psychophysics
amoeba_struct.target_outer_max = 0.5;%max/min outer radius of target annulus, units of image rect
amoeba_struct.target_outer_min = 0.75; % 0.5 value in Geisler paper
amoeba_struct.target_inner_max = 0.5;%max/min inner radius in units of outer radius
amoeba_struct.target_inner_min = 0.75; % 0.5 value in Geisler paper
amoeba_struct.num_phi = 1024;
amoeba_struct.num_fourier = nfourier; %min(3,amoeba_struct.num_phi);
amoeba_struct.min_gap = 16; % 32; % 
amoeba_struct.max_gap = 32; % 64; % 
amoeba_struct.fourier_amp = zeros(amoeba_struct.num_fourier, 1);
% set amp of largest fourier component factor of 2 larger to make more distinct amoebas
				% amoeba_struct.fourier_amp(amoeba_struct.num_fourier,1) = 1;

if isempty(mean_amoeba_xy)
  num_objects = amoeba_struct.num_targets + amoeba_struct.num_distractors;
  mean_amoeba_xy = zeros(num_trials*amoeba_struct.num_targets,2);
  std_amoeba_xy = zeros(num_trials*amoeba_struct.num_targets,2);
  mean_distractor_xy = zeros(num_trials*(2*num_objects - amoeba_struct.num_targets),2);
  std_distractor_xy = zeros(num_trials*(2*num_objects - amoeba_struct.num_targets),2);
endif

[amoeba_image] = amoeba2D2(amoeba_struct);
plotAmoeba2D2(amoeba_image,  amoeba_struct, trial_ndx, nfourier);


