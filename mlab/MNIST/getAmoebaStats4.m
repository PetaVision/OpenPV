function getAmoebaStats4(trial_ndx, target_id, target_type, ...
			 num_fourier, num_trials)

  if nargin < 3 || ~exist("target_type") || isempty(target_type)
    target_type = 0; %% radial frequency pattern + clutter
    %% target_type = 1; %% MNIST digits (resized) + clutter
  endif
  if nargin < 2 || ~exist("target_id") || isempty(target_id)
    if target_type == 0
      target_id = 4;
    elseif target_type == 1
      target_id = 0;
    endif
  endif
  if nargin < 4 || ~exist("num_fourier") || isempty(num_fourier)
    if target_type == 0
      num_fourier = target_id;
    elseif target_type == 1
      num_fourier = 4;
    endif
  endif
  if nargin < 5 || ~exist("num_trials") || isempty(num_trials)
    num_trials = 1;
  endif
 

  global image_dim
  global mean_amoeba_xy std_amoeba_xy mean_distractor_xy std_distractor_xy
  %%global j_trial num_trials
  j_trial = trial_ndx;

  
  amoeba_struct = struct;
				%rand('twister', sum(100*clock));  % rand automatically initialize "randomly"
				%amoeba_struct.rand_state = {rand('twister')};
  amoeba_struct.rand_state = {rand('state')};
  amoeba_struct.num_segments = 2^4; %%
  amoeba_struct.image_rect_size = image_dim(1);

  amoeba_struct.num_targets = 1;
  amoeba_struct.num_distractors = 2; %in addition to amoeba targets, total
				% objects is sum of both
  amoeba_struct.segments_per_distractor = 2^(-4); % %as fraction
				% of num_segments, 2^(-2)  used in psychophysics
  amoeba_struct.target_outer_max = 0.5;%max/min outer radius of target annulus, units of image rect
  amoeba_struct.target_outer_min = 0.5; %% value in Geisler paper
  amoeba_struct.target_inner_max = 0.5;%max/min inner radius in units of outer radius
  amoeba_struct.target_inner_min = 0.5; %% value in Geisler paper
  amoeba_struct.num_fourier = num_fourier; %min(3,amoeba_struct.num_phi);
  amoeba_struct.min_gap = 16; % 32; % 
  amoeba_struct.max_gap = 32; % 64; % 
  amoeba_struct.fourier_amp = zeros(amoeba_struct.num_fourier, 1);
				% set amp of largest fourier component factor of 2 larger to make more distinct amoebas
				% amoeba_struct.fourier_amp(amoeba_struct.num_fourier,1) = 1;
  amoeba_struct.min_resize = 1; %% use 2 for MNIST
  amoeba_struct.max_resize = 1; %% use 5 for MNIST
  amoeba_struct.closed_prob = 1;  %% prob of closed vs open(linear) amoeba/clutter, use 0.5 for MNISt
  amoeba_struct.num_phi = 1024;
  amoeba_struct.range_phi = 2*pi; %% < 2*pi generates open figures

  amoeba_struct.target_id = target_id;
  amoeba_struct.target_type = target_type;

if isempty(mean_amoeba_xy)
  num_objects = amoeba_struct.num_targets + amoeba_struct.num_distractors;
  mean_amoeba_xy = zeros(num_trials*amoeba_struct.num_targets,2);
  std_amoeba_xy = zeros(num_trials*amoeba_struct.num_targets,2);
  mean_distractor_xy = zeros(num_trials*(2*num_objects - amoeba_struct.num_targets),2);
  std_distractor_xy = zeros(num_trials*(2*num_objects - amoeba_struct.num_targets),2);
endif

[amoeba_image] = amoeba2D3(amoeba_struct, trial_ndx);
plotAmoeba2D3(amoeba_image,  amoeba_struct, trial_ndx);


