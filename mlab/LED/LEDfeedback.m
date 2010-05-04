% Psycho Physics LED experiment
% We have two LED patterns (left and right)
% One contains a digit between 0->9, the other a distractor letter or non-semantic shape.
% We have to decide which patterns contains the digit.


% define left right patterns and Mask (M)
% Segments ordering:
%      1
%     ---
%    |2  |3
%      4
%     ---
%    | 5  |6
%      7
%     ---

if ( uioctave )
  setenv("GNUTERM", "x11");
endif

clear all
close all

%% plotting flags

plot_LED_patterns = 1;
plot_performance = 1;

% ploting symbols
global plot_marker
plot_marker = { 'o', 'x', '*', '^', 'd', 's', '+'};

%% maximum stimulus onset asynchrony (== duration of image presentation == time
% of mask presentation)
max_SoA = 160;  % msec

global LED_segment_prior
LED_segment_prior = 0.5;

global LED_detector_prior
LED_detector_prior = 0.5;

%% define LED targets and distractors
global num_LED_segs
num_LED_segs = 7;

global LED_patterns
[LED_patterns, ...
    LED_NUMBER_ndx, LED_NUMBER_str, num_LED_NUMBERS, ...
    LED_LETTER_ndx, LED_LETTER_str, num_LED_LETTERS, ...
    LED_RANDOM_ndx, LED_RANDOM_str, num_LED_RANDOM, ...
    LED_BLANK_ndx, LED_DASH_ndx, ...
    LED_SEMANTIC_ndx, num_LED_SEMANTIC, LED_NONSEMANTIC_ndx, num_LED_NONSEMANTIC] = ...
    LEDPatterns();

%% plot LED patterns
if plot_LED_patterns
    plotLEDPatternTableaux(LED_patterns, LED_SEMANTIC_ndx, num_LED_SEMANTIC, 'LED: Semantic (Letters and Numbers)');
    plotLEDPatternTableaux(LED_patterns, LED_NONSEMANTIC_ndx, num_LED_NONSEMANTIC, 'LED: Non-Semantic');
end

time_steps = 1 : ceil(1.5 * max_SoA);
max_time = max(time_steps);

%% load experimental data

data_SoA = [ 20, 40, 80, 160 ];

% data_GTK = zeros( [2, 6, 4] );

% data_GTK(1,:,:) = [
%     0.7500    0.9167    0.5417    0.5417    0.5417    0.7083
%     1.0000    1.0000    0.7917    1.0000    0.8333    0.7391
%     1.0000    1.0000    0.9583    0.9167    1.0000    0.9583
%     1.0000    1.0000    1.0000    1.0000    1.0000    1.0000
%     ]';

% data_GTK(2,:,:) = [
%     0.9167    0.5833    0.5000    0.4583    0.4583    0.5000
%     0.9583    0.7917    0.7500    0.7083    0.6250    0.5833
%     1.0000    1.0000    1.0000    0.9167    0.9583    1.0000
%     1.0000    1.0000    1.0000    0.9583    1.0000    1.0000
%     ]';

% first dimension is SOA;  second is the mask.
% third is the target (2,4).  fourth is the
% distractor (E,h), fifth is the participant.
load results4;

allsubjects_data = results4;
[num_SoAs_data, num_masks_data, num_targets_data, num_distractors_data, num_subjects] = size( allsubjects_data);

subjects_list = 2; %1:num_subjects;%
num_subjects = length(subjects_list);

%% initialize targets/distractors/masks

% targets
% 2, 4
  global start_target stop_target
num_targets = 2; % num_targets_data;
start_target = 2;
stop_target = 2;
target_ndx_list = [ LED_NUMBER_ndx(2), LED_NUMBER_ndx(4) ]; %
target_str_list = { LED_NUMBER_str{2}, LED_NUMBER_str{4} }; %

% distractors
% h, E
  global start_distractor stop_distractor
num_distractors = 2; % num_distractors_data;
start_distractor = 2;
stop_distractor = 2;
distractor_ndx_list = [ LED_LETTER_ndx(16), LED_LETTER_ndx(3) ]; %
distractor_str_list = { LED_LETTER_str{16}, LED_LETTER_str{3} }; %

% masks
% 1, 7, 5 - center segment, 5, A, 8 (in order of # of segments)
mask_ndx_list = [ LED_NUMBER_ndx(1) LED_NUMBER_ndx(7) LED_RANDOM_ndx(10) LED_NUMBER_ndx(5) LED_LETTER_ndx(1) LED_NUMBER_ndx(8) ];
mask_str_list = { LED_NUMBER_str{1} LED_NUMBER_str{7} LED_RANDOM_str{10} LED_NUMBER_str{5} LED_LETTER_str{1} LED_NUMBER_str{8} };
num_masks = length(mask_ndx_list(1:num_masks_data));


%% plot experimental data


tau_segment = 10.0; % msec
tau_detector = 20.0;  % msec
kappa_struct = struct;  % LED_segment >= 0.0 -> ON, otherwise, OFF
kappa_struct.feedforward_present = 1.0*1.0; % weight to pattern detector from LED segment "present" in pattern
kappa_struct.feedforward_absent = -1.0*4.0; % weight to pattern detector from LED segment "absent" in pattern
kappa_struct.feedback_present = 1.0*0.75; % weight from pattern detector to LED segment "present" in pattern
kappa_struct.feedback_absent = -1.0*0.25; % weight from pattern detector to LED segment "absent" in pattern
kappa_struct.ON_OFF_symmetry = 0.5;
kappa_struct.OFF_OFF_symmetry = 0.0;
kappa_struct.winner_take_all_inhib = -1.0*4.0;
kappa_struct.gain = 5.0;
semantic_flag = 1;  % == 1 if mask is semantic, 0 otherwise

LED_state_init = zeros(num_LED_segs + 3,1);
LED_state_init(1:num_LED_segs,1) = 0;
LED_state_init(num_LED_segs+1:end) = 0;

% target variables
LED_target_segment = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_target  = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_target_mask = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_target_distractor = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
% distractor variables
LED_distractor_segment = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_distractor  = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_distractor_mask = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
LED_distractor_target = cell(num_targets, num_distractors, num_masks, num_SoAs_data);
% performance variables
LED_prob_correct = cell(num_targets, num_distractors, num_masks);

for i_mask = 1 : num_masks
    mask_ndx = mask_ndx_list(i_mask);
    mask_str = mask_str_list{i_mask};
    mask_pattern = LED_patterns(mask_ndx, :);

    for i_target = start_target : stop_target
        target_ndx = target_ndx_list(i_target);
        target_str = target_str_list{i_target};
        target_pattern = LED_patterns(target_ndx, :);

        for i_distractor = start_distractor : stop_distractor
            distractor_ndx = distractor_ndx_list(i_distractor);
            distractor_str = distractor_str_list{i_distractor};
            distractor_pattern = LED_patterns(distractor_ndx, :);

            for i_SoA = 1 : num_SoAs_data
                time_SoA = data_SoA(i_SoA);
                [ LED_steps, LED_state ] = ...
                    ode45( @(time_steps, LED_state_init) ...
                    LEDtopdown( ...
                    time_steps, ...
                    LED_state_init, ...
                    target_pattern, ...
                    mask_pattern, ...
                    target_pattern, ...
                    distractor_pattern, ...
                    time_SoA, ...
                    tau_segment, ...
                    tau_detector, ...
                    kappa_struct, ...
                    semantic_flag ), ...
                    time_steps, ...
                    LED_state_init );

                LED_target_segment_tmp = LED_state(:, 1:num_LED_segs);
                LED_target_tmp = LED_state(:, num_LED_segs + 1);
                LED_target_mask_tmp = LED_state(:, num_LED_segs + 2);
                LED_target_distractor_tmp = LED_state(:, num_LED_segs + 3);
                LED_target_segment{i_target, i_distractor, i_mask, i_SoA} = LED_target_segment_tmp;
                LED_target{i_target, i_distractor, i_mask, i_SoA} = LED_target_tmp;
                LED_target_mask{i_target, i_distractor, i_mask, i_SoA} = LED_target_mask_tmp;
                LED_target_distractor{i_target, i_distractor, i_mask, i_SoA} = LED_target_distractor_tmp;

            end % time_SoA

            for i_SoA = 1 : num_SoAs_data
                time_SoA = data_SoA(i_SoA);
                [ LED_steps, LED_state ] = ...
                    ode45( @(time_steps, LED_state_init) ...
                    LEDtopdown( ...
                    time_steps, ...
                    LED_state_init, ...
                    distractor_pattern, ...
                    mask_pattern, ...
                    target_pattern, ...
                    distractor_pattern, ...
                    time_SoA, ...
                    tau_segment, ...
                    tau_detector, ...
                    kappa_struct, ...
                    semantic_flag ), ...
                    time_steps, ...
                    LED_state_init );

                LED_distractor_segment_tmp = LED_state(:, 1:num_LED_segs);
                LED_distractor_target_tmp = LED_state(:, num_LED_segs + 1);
                LED_distractor_mask_tmp = LED_state(:, num_LED_segs + 2);
                LED_distractor_tmp = LED_state(:, num_LED_segs + 3);
                LED_distractor_segment{i_target, i_distractor, i_mask, i_SoA} = LED_distractor_segment_tmp;
                LED_distractor_target{i_target, i_distractor, i_mask, i_SoA} = LED_distractor_target_tmp;
                LED_distractor_mask{i_target, i_distractor, i_mask, i_SoA} = LED_distractor_mask_tmp;
                LED_distractor{i_target, i_distractor, i_mask, i_SoA} = LED_distractor_tmp;

            end % time_SoA
        end % i_distractor
    end % i_target
end % i_mask

for i_mask = 1 : num_masks
    for i_target = start_target : stop_target
        for i_distractor = start_distractor : stop_distractor
            LED_prob_correct{i_target, i_distractor, i_mask} = repmat(0.5, num_SoAs_data, 1);
            for i_SoA = 1 : num_SoAs_data
	        first_ndx = find( LED_target{i_target, i_distractor, i_mask, i_SoA} > 0.25, 1, 'first' );
		if isempty( first_ndx)
		  first_ndx = max_time;
		end
	        last_ndx = find( LED_target{i_target, i_distractor, i_mask, i_SoA} > 0.25, 1, 'last' );
		if isempty(last_ndx)
		  last_ndx = 0;
		end
		numerator1 = min( max( ( last_ndx - first_ndx + 1 ) / 100, 0.0 ), 1.0 );
	        first_ndx2 = find( LED_distractor_target{i_target, i_distractor, i_mask, i_SoA} > 0.25, 1, 'first' );
		if isempty( first_ndx2)
		  first_ndx2 = max_time;
		end
	        last_ndx2 = find( LED_distractor_target{i_target, i_distractor, i_mask, i_SoA} > 0.25, 1, 'last' );
		if isempty(last_ndx2)
		  last_ndx2 = 0;
		end
		numerator2 = min( max( ( last_ndx2 - first_ndx2 + 1 ) / 100, 0.0 ), 1.0 );
		numerator = 0.5 + 0.5 * ( numerator1 - numerator2 );
%                numerator = 0.5 + 0.5 * ...
%                    ( min( max( LED_target{i_target, i_distractor, i_mask, i_SoA} ), 1.0 ) - ...
%                    min( max( LED_distractor_target{i_target, i_distractor, i_mask, i_SoA} ), 1.0 ) );
                denominator = 1;
                if denominator == 0
                    denominator = 1;
                end
                LED_prob_correct{i_target, i_distractor, i_mask}(i_SoA, 1) = ...
                    numerator / denominator;
            end % time_SoA
        end % i_distractor
    end % i_target
end % i_mask

%% plot results
if plot_performance
    fh_performance = ...
        LEDplotSubjectsPerformance( ...
        num_targets, target_str_list, ...
        num_distractors, distractor_str_list, ...
        num_masks, ...
        num_subjects, subjects_list, allsubjects_data, data_SoA, max_SoA );

    fh_performance = ...
        LEDplotModelPerformance( num_targets, ...
        num_distractors, ...
        num_masks, mask_str_list, ...
        num_subjects, subjects_list, LED_prob_correct, data_SoA, max_SoA, fh_performance );
end % plot_performance

plot_detectors = 1;
if plot_detectors
    fh_detector = ...
        LEDplotDetectors( num_targets, target_str_list, ...
        num_distractors, distractor_str_list, ...
        num_masks, mask_str_list, ...
        data_SoA, num_SoAs_data, max_time, ...
        LED_target, LED_distractor_target, ...
        LED_target_mask, LED_distractor_mask, ...
        LED_distractor, LED_target_distractor);
end % plot_detectors


plot_segments = 1;  % leave off for now...
if plot_segments
    fh_segments = ...
        LEDplotSegments( num_targets, target_str_list, ...
        num_distractors, distractor_str_list, ...
        num_masks, mask_str_list, ...
        data_SoA, num_SoAs_data, max_time, ...
        LED_target_segment, LED_distractor_segment);
end % plot_segments


