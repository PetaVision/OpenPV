function getAmoebaStats2(t, nfour) 

amoeba_struct = struct;
rand('twister', sum(100*clock));
amoeba_struct.rand_state = {rand('twister')};
amoeba_struct.num_segments = 2^4;
amoeba_struct.image_rect_size = 128; %256;



amoeba_struct.num_targets = 1;
%amoeba_struct.num_distractors = 0*amoeba_struct.num_targets; %for targets
amoeba_struct.num_distractors = 5; %for no targets
amoeba_struct.segments_per_distractor = 2^(-3);  %as fraction of num_segments
amoeba_struct.target_outer_max = 0.5;%1; %max/min outer radius of target annulus, units of image rect
amoeba_struct.target_outer_min = 0.5;%0.5;%1.0;%
amoeba_struct.target_inner_max = 0.5;%0.75;%1.0;% %max/min inner radius in units of outer radius
amoeba_struct.target_inner_min = 0.5;%0.25;%1.0;%
amoeba_struct.num_phi = 1024;
amoeba_struct.num_fourier = nfour; %min(3,amoeba_struct.num_phi);
amoeba_struct.min_gap = 16;
amoeba_struct.max_gap = 32;

get_stats = 0;
num_trials = 1;
cum_intra = [];
cum_inter = [];
tot_num_intra = 0;
tot_num_inter = 0;
plot_amoeba2D = 1;


[amoeba_image] = amoeba2D2(amoeba_struct);
plotAmoeba2D(amoeba_image,  t, nfour);


