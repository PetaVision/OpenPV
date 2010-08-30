function getAmoebaStats3(t, nfour)

  global image_dim

amoeba_struct = struct;
%rand('twister', sum(100*clock));
%amoeba_struct.rand_state = {rand('twister')};
amoeba_struct.rand_state = {rand('state')};
amoeba_struct.num_segments = 2^4; % 2^3; % 
amoeba_struct.image_rect_size = image_dim(1);



amoeba_struct.num_targets = 1;
%amoeba_struct.num_distractors = 0*amoeba_struct.num_targets; %for targets
amoeba_struct.num_distractors = 5; %for no targets
amoeba_struct.segments_per_distractor =  2^(-3); % 2^(-2); % %as fraction
				% of num_segments, 2nd value used in psychophysics
amoeba_struct.target_outer_max = 0.5;%1; %max/min outer radius of target annulus, units of image rect
amoeba_struct.target_outer_min = 0.5;%0.5;%1.0;%
amoeba_struct.target_inner_max = 0.5;%0.75;%1.0;% %max/min inner radius in units of outer radius
amoeba_struct.target_inner_min = 0.5;%0.25;%1.0;%
amoeba_struct.num_phi = 1024;
amoeba_struct.num_fourier = nfour; %min(3,amoeba_struct.num_phi);
amoeba_struct.min_gap = 16; % 32; % 
amoeba_struct.max_gap = 32; % 64; % 

[amoeba_image] = amoeba2D2(amoeba_struct);
plotAmoeba2D2(amoeba_image,  t, nfour);


