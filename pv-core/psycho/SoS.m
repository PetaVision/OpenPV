clear all
SoS_seed = sum(100*clock);
rand('state', SoS_seed); % reseed the random-number generator for each expt.
SoS_state = rand('state');

try
    %% init window
    Screen('Preference','SkipSyncTests',0); %[1] to skip screen tests
    [SoS_w0, SoS_w0_rect] = Screen('OpenWindow',0);
    disp(['SoS_w0_rect = ', num2str(SoS_w0_rect)]);
    %Screen('Close', SoS_w0);
    SoS_x_1_2 = round(SoS_w0_rect(3)/2); %horizontal center
    SoS_y_1_2 = round(SoS_w0_rect(4)/2); %vertical center
    disp(['SoS_x_1_2 = ', num2str(SoS_x_1_2)]);
    disp(['SoS_y_1_2 = ', num2str(SoS_y_1_2)]);
    SoS_x_1_4 = round(SoS_x_1_2 / 2); % other useful screen landmarks
    SoS_y_1_4 = round(SoS_y_1_2 / 2);
    SoS_x_3_4 = round(3 * SoS_x_1_4 );
    SoS_y_3_4 = round(3 * SoS_y_1_4 );
    SoS_x_1_8 = round(SoS_x_1_4 / 2);
    SoS_y_1_8 = round(SoS_y_1_4 / 2);
    SoS_x_7_8 = round(7 * SoS_x_1_8 );
    SoS_y_7_8 = round(7 * SoS_y_1_8 );
    SoS_foreground_height = SoS_y_7_8 - SoS_y_1_8 + 1;
    SoS_foreground_height = 2^( round( log2( SoS_foreground_height ) ) );
    SoS_foreground_width = SoS_foreground_height;
    SoS_foreground_left = SoS_x_1_2 - ( SoS_foreground_width / 2 );
    SoS_foreground_bottom = SoS_y_1_2 - ( SoS_foreground_height / 2 );
    SoS_foreground_right = SoS_x_1_2 + ( SoS_foreground_width / 2 ) - 1;
    SoS_foreground_top = SoS_y_1_2 + ( SoS_foreground_height / 2 ) - 1;
    SoS_foreground_rect = ...
        [SoS_foreground_left SoS_foreground_bottom SoS_foreground_right SoS_foreground_top];
    grayCLUTndx = GrayIndex(SoS_w0);
    whiteCLUTndx = WhiteIndex(SoS_w0);
    blackCLUTndx = BlackIndex(SoS_w0);
    redCLUTndx = [whiteCLUTndx(1) 0 0];
    greenCLUTndx = [0 whiteCLUTndx(1) 0];
    blueCLUTndx = [0 0 whiteCLUTndx(1)];
    hz = Screen('FrameRate',SoS_w0);
    frame_duration = Screen('GetFlipInterval', SoS_w0);
    frame_rate = 1/frame_duration;
    disp(['hz = ', num2str(hz)]);
    disp(['frame_duration = ', num2str(frame_duration)]);
    disp(['frame_rate = ', num2str(frame_rate)]);

    %% init text params
    default_text_size = 36;
    text_style_str = {'normal','bold','italic','underline','outline','condense','extend' };
    old_text_size = Screen('TextSize', SoS_w0, default_text_size);
    old_text_font = Screen('TextFont', SoS_w0);
    old_text_style = Screen('TextStyle', SoS_w0);

    %% init choice params
    [instr_text, tmp] = ...
        sprintf('%6s\n', ...
        'Options:', ...
        'up arrow: target present', ...
        'down arrow: target absent', ...
        'right arrow: discard trial', ...
        'esc once to pause, twice to exit');
    %         'left arrow: switch previous choice and discard current trial', ...
    choice_text_rect = RectWidth( Screen( 'TextBounds', SoS_w0, instr_text ) );
    escape_key = KbName('esc');
    up_key = KbName('up');
    down_key = KbName('down');
    left_key = KbName('left');
    right_key = KbName('right');
    enter_key = KbName('return');
    end_key = KbName('end');
    home_key = KbName('home');
    exit_experiment = 0;


    %% init target and clutter params
    SoS_num_trials = 20; %number of trials for each condition
    SoS_file_new = ['SoSExp', num2str(round(SoS_seed)), '.mat'];
    SoS_files = {SoS_file_new};
    SoS_scale = 1/16;%1/1;% reduction from screen pixels to image pixels
    SoS_height = SoS_scale * SoS_foreground_height;
    SoS_width = SoS_scale * SoS_foreground_width;
    SoS_num_theta = 8;
    SoS_delta_theta = 180 / SoS_num_theta;
    SoS_num_duration_vals = 4; %number of durations per trial
    SoS_duration_min = 1.0/75; %min/max viewing duration
    SoS_duration_max = 5.0/75;
    SoS_ITI = 1; %min/max inter-trial-interval
    SoS_radius_min = 0.1; %min/max target radius (fraction of foreground rectangle)
    SoS_radius_max = 0.2;
    SoS_offset_min = -0.1; %min/max target center (fraction of foreground rectangle)
    SoS_offset_max = 0.1;
    SoS_num_target_vals = 1;
    SoS_target_min = 1.0;  %max/min percent target segments
    SoS_target_max = 1.0;
    SoS_num_clutter_vals = 2;
    SoS_clutter_min = 0.1;  %min/max percent clutter
    SoS_clutter_max = 0.25;
    SoS_gray_mid = mean(grayCLUTndx);
    SoS_gray_min = blackCLUTndx;%
    SoS_gray_max = whiteCLUTndx;%
    SoS_skip_flag = 10; %trial skipped
    SoS_fixnum_flag = 0; %1: fix number of segments to maintain constant luminance
    SoS_shuffle_flag = 1; %flag to control whether targets are shuffled/scrambled
    SoS_tot_combinations = SoS_num_duration_vals * SoS_num_target_vals * SoS_num_clutter_vals;
    SoS_tot_trials = SoS_num_trials * SoS_tot_combinations;
    SoS_angle = 0; %degrees
    SoS_duration = zeros(SoS_tot_trials,1);
    SoS_radius_x = zeros(SoS_tot_trials,1);
    SoS_radius_y = zeros(SoS_tot_trials,1);
    SoS_offset_x = zeros(SoS_tot_trials,1);
    SoS_offset_y = zeros(SoS_tot_trials,1);
    SoS_target = zeros(SoS_tot_trials,1);
    SoS_clutter = zeros(SoS_tot_trials,1);
    SoS_choice = repmat( SoS_skip_flag , SoS_tot_trials, 1 );
    SoS_notarget = ones(SoS_tot_trials,1);

    SoS_trial_index = repmat( 1 : SoS_tot_combinations, SoS_num_trials, 1 );
    SoS_trial_index = reshape( SoS_trial_index, SoS_tot_trials, 1 );
    if SoS_shuffle_flag == 1
        [SoS_trial_index, SoS_shuffle_index] = Shuffle(SoS_trial_index);
    end
    SoS_duration_delta = ( SoS_duration_max - SoS_duration_min + ( SoS_num_duration_vals <= 1 ) ) / ...
        ( SoS_num_duration_vals - 1 + ( SoS_num_duration_vals <= 1 ) ); %bin width for analysis
    SoS_target_delta = ( SoS_target_max - SoS_target_min + ( SoS_num_target_vals <= 1 ) ) / ...
        ( SoS_num_target_vals - 1 + ( SoS_num_target_vals <= 1 ) ); %bin width for analysis
    SoS_clutter_delta = (SoS_clutter_max - SoS_clutter_min + ( SoS_num_clutter_vals <= 1 ) ) / ...
        ( SoS_num_clutter_vals - 1 + ( SoS_num_clutter_vals <= 1 ) ); %bin width for analysis
    SoS_duration_vals = ...
        SoS_duration_min : SoS_duration_delta : SoS_duration_max;
    SoS_target_vals = ...
        SoS_target_min : SoS_target_delta : SoS_target_max;
    SoS_clutter_vals = ...
        SoS_clutter_min : SoS_clutter_delta : SoS_clutter_max;

    SoS_data = struct( ...
        'SoS_seed', SoS_seed, ...
        'SoS_state', SoS_state, ...
        'SoS_scale', SoS_scale, ...
        'SoS_height', SoS_height, ...
        'SoS_width', SoS_width, ...
        'SoS_num_theta', SoS_num_theta, ...
        'SoS_ITI', SoS_ITI, ...
        'SoS_radius_min', SoS_radius_min, ...
        'SoS_radius_max', SoS_radius_max, ...
        'SoS_radius_x', SoS_radius_x, ...
        'SoS_radius_y', SoS_radius_y, ...
        'SoS_offset_min', SoS_offset_min, ...
        'SoS_offset_max', SoS_offset_max, ...
        'SoS_offset_x', SoS_offset_x, ...
        'SoS_offset_y', SoS_offset_y, ...
        'SoS_duration_min', SoS_duration_min, ...
        'SoS_duration_max', SoS_duration_max, ...
        'SoS_duration', SoS_duration, ...
        'SoS_num_duration_vals', SoS_num_duration_vals, ...
        'SoS_duration_delta', SoS_duration_delta, ...
        'SoS_duration_vals', SoS_duration_vals, ...
        'SoS_target_min', SoS_target_min, ...
        'SoS_target_max', SoS_target_max, ...
        'SoS_target', SoS_target, ...
        'SoS_num_target_vals', SoS_num_target_vals, ...
        'SoS_target_delta', SoS_target_delta, ...
        'SoS_target_vals', SoS_target_vals, ...
        'SoS_clutter_min', SoS_clutter_min, ...
        'SoS_clutter_max', SoS_clutter_max, ...
        'SoS_clutter', SoS_clutter, ...
        'SoS_num_clutter_vals', SoS_num_clutter_vals, ...
        'SoS_clutter_delta', SoS_clutter_delta, ...
        'SoS_clutter_vals', SoS_clutter_vals, ...
        'SoS_gray_mid', SoS_gray_mid, ...
        'SoS_gray_min', SoS_gray_min, ...
        'SoS_gray_max', SoS_gray_max, ...
        'SoS_skip_flag', SoS_skip_flag, ...
        'SoS_fixnum_flag', SoS_fixnum_flag, ...
        'SoS_shuffle_flag', SoS_shuffle_flag, ...
        'SoS_choice', SoS_choice, ...
        'SoS_notarget', SoS_notarget, ...
        'SoS_angle', SoS_angle, ...
        'SoS_files', SoS_files ...
        );


    %set background color
    Screen('FillRect', SoS_w0, SoS_gray_mid);
    Screen('Flip', SoS_w0);

    %% start experiment
    SoS_correct_tally = 0;
    SoS_incorrect_tally = 0;
    SoS_num_edges = SoS_height * SoS_width * SoS_num_theta;
    SoS_I = repmat(SoS_gray_mid, [SoS_height, SoS_width, SoS_num_theta] );
    SoS_edge_len = 16*sqrt(2);
    SoS_penWidth = 3;
    for SoS_trial = 1 : SoS_tot_trials
        [SoS_duration_ndx, SoS_target_ndx, SoS_clutter_ndx] = ...
            ind2sub([SoS_num_duration_vals, SoS_num_target_vals, SoS_num_clutter_vals], ...
            SoS_trial_index(SoS_trial));
        SoS_duration(SoS_trial) = SoS_duration_vals(SoS_duration_ndx);
        SoS_clutter(SoS_trial) = SoS_clutter_vals(SoS_clutter_ndx);
        Screen('DrawText',SoS_w0, 'rendering image...', SoS_x_1_2, SoS_y_1_2, blackCLUTndx);
        Screen('Flip', SoS_w0);
        %make target (dimensions are relative to foreground rect)
        if rand > 0.5
            SoS_target(SoS_trial) = SoS_target_vals(SoS_target_ndx);
            SoS_notarget(SoS_trial) = 0;
            SoS_radius_x(SoS_trial) = ...
                fix( ( SoS_radius_min + rand * (SoS_radius_max - SoS_radius_min) ) * SoS_foreground_width );
            SoS_radius_y(SoS_trial) = ...
                fix( ( SoS_radius_min + rand * (SoS_radius_max - SoS_radius_min) ) * SoS_foreground_height );
            SoS_offset_x(SoS_trial) = ...
                fix( ( SoS_offset_min + rand * (SoS_offset_max - SoS_offset_min) ) * SoS_foreground_width );
            SoS_offset_y(SoS_trial) = ...
                fix( ( SoS_offset_min + rand * (SoS_offset_max - SoS_offset_min) ) * SoS_foreground_height );
            SoS_target_rect_left = SoS_x_1_2 - SoS_radius_x(SoS_trial) + SoS_offset_x(SoS_trial);
            SoS_target_rect_top = SoS_y_1_2 + SoS_radius_y(SoS_trial) + SoS_offset_y(SoS_trial);
            SoS_target_rect_right = SoS_x_1_2 + SoS_radius_x(SoS_trial) + SoS_offset_x(SoS_trial);
            SoS_target_rect_bottom = SoS_y_1_2 - SoS_radius_y(SoS_trial) + SoS_offset_y(SoS_trial);
            SoS_target_rect = ...
                [SoS_target_rect_left SoS_target_rect_bottom ...
                SoS_target_rect_right SoS_target_rect_top];
            Screen( 'FrameOval', SoS_w0, SoS_gray_max, SoS_target_rect );
        end % rand
        % make clutter
        SoS_num_clutter = round( SoS_num_edges * SoS_clutter(SoS_trial) );
        SoS_clutter_lines = zeros(2, 2*SoS_num_clutter);
        SoS_theta = rand(SoS_num_clutter,1) * pi;
        SoS_x = ( rand(SoS_num_clutter,1) - 0.5 ) * SoS_foreground_width;
        SoS_y = ( rand(SoS_num_clutter,1) - 0.5 ) * SoS_foreground_height;
        SoS_delta_x = SoS_edge_len * abs(cos(SoS_theta));
        SoS_delta_y = SoS_edge_len * abs(sin(SoS_theta));
        SoS_clutter_lines( 1, 1:2:2*SoS_num_clutter) = SoS_x - SoS_delta_x;
        SoS_clutter_lines( 2, 1:2:2*SoS_num_clutter) = SoS_y - SoS_delta_y;
        SoS_clutter_lines( 1, 2:2:2*SoS_num_clutter) = SoS_x + SoS_delta_x;
        SoS_clutter_lines( 2, 2:2:2*SoS_num_clutter) = SoS_y + SoS_delta_y;
        Screen('DrawLines', SoS_w0, SoS_clutter_lines, SoS_penWidth, SoS_gray_max, [SoS_x_1_2, SoS_y_1_2]);
%         [SoS_I_clutter, SoS_clutter_ndx] = sort(rand( SoS_num_edges , 1));
%         [SoS_x_i, SoS_y_i, SoS_theta_i] = ...
%             ind2sub([SoS_height, SoS_width, SoS_num_theta], ...
%             SoS_clutter_ndx);
%         for SoS_line = 1 : SoS_num_clutter
%             [SoS_x_i, SoS_y_i, SoS_theta_i] = ...
%                 ind2sub([SoS_height, SoS_width, SoS_num_theta], ...
%                 SoS_clutter_ndx(SoS_line));
%             SoS_theta = ( SoS_theta_i - 1 ) * SoS_delta_theta * pi / 180;
%             SoS_x = SoS_foreground_left + 0.5 + ( SoS_x_i - 0.5 ) / SoS_scale;
%             SoS_y = SoS_foreground_bottom + 0.5 + ( SoS_y_i - 0.5 ) / SoS_scale;
%             SoS_delta_x = ( SoS_edge_len * abs(cos(SoS_theta)) ) / SoS_scale;
%             SoS_delta_y = ( SoS_edge_len * abs(sin(SoS_theta)) ) / SoS_scale;
%             SoS_clutter_left = ( SoS_x - SoS_delta_x );
%             SoS_clutter_bottom = ( SoS_y - SoS_delta_y );
%             SoS_clutter_right = ( SoS_x + SoS_delta_x );
%             SoS_clutter_top = ( SoS_y + SoS_delta_y );
%             SoS_clutter_lines( 1, 2*SoS_line-1) = SoS_clutter_left;
%             SoS_clutter_lines( 2, 2*SoS_line-1) = SoS_clutter_bottom;
%             SoS_clutter_lines( 1, 2*SoS_line) = SoS_clutter_right;
%             SoS_clutter_lines( 2, 2*SoS_line) = SoS_clutter_top;
%             Screen('DrawLine', SoS_w0, SoS_gray_max, ...
%                 SoS_clutter_left, SoS_clutter_bottom, SoS_clutter_right, SoS_clutter_top, ...
%                 SoS_penWidth);
%         end
%         Screen('Flip', SoS_w1, 0, 1);  %use to view rendered image and bypass experiment 
%         WaitSecs(2);
%         Screen('CloseAll');
%         break;
        max_priority = MaxPriority(SoS_w0, 'WaitSecs');
        SoS_stim = {
            'WaitSecs(SoS_ITI);'
            'Screen(''Flip'', SoS_w0);'
            'WaitSecs(SoS_duration(SoS_trial));'
            'Screen(''FillRect'', SoS_w0, SoS_gray_mid);'
            'Screen(''Flip'', SoS_w0);'
            };
        Rush( SoS_stim, max_priority );

        %parse user keybord input (note some options are nested to allow for pausing experiment)
        while KbCheck; end  %clears keyboard buffer from previous trial
        while 1
            [ keyIsDown, seconds, keyCode ] = KbCheck;
            WaitSecs(0.001) % delay to prevent CPU hogging
            if keyIsDown
                if keyCode(up_key) % taget present
                    SoS_choice(SoS_trial) = 1;
                    if SoS_notarget(SoS_trial) == 1
                        SoS_incorrect_tally = SoS_incorrect_tally + 1;
                        for i=1:3
                            beep;pause(0.5);
                        end
                    else
                        SoS_correct_tally = SoS_correct_tally + 1;
                    end
                    tally_str = ['% correct = ',num2str(SoS_correct_tally/(SoS_correct_tally+SoS_incorrect_tally))];
                    Screen('DrawText',SoS_w0, tally_str, SoS_x_1_2, SoS_y_1_2, blackCLUTndx);
                    Screen('Flip', SoS_w0);
                    pause(0.5)
                    break;
                elseif keyCode(down_key) % target absent
                    SoS_choice(SoS_trial) = -1;
                    if SoS_notarget(SoS_trial) == 0
                        SoS_incorrect_tally = SoS_incorrect_tally + 1;
                        for i=1:3
                            beep;pause(0.5);
                        end
                    else
                        SoS_correct_tally = SoS_correct_tally + 1;
                    end
                    tally_str = ['% correct = ',num2str(SoS_correct_tally/(SoS_correct_tally+SoS_incorrect_tally))];
                    Screen('DrawText',SoS_w0, tally_str, SoS_x_1_2, SoS_y_1_2, blackCLUTndx);
                    Screen('Flip', SoS_w0);
                    pause(0.5)
                    break;
                elseif keyCode(right_key) %skip this trial
                    SoS_choice(SoS_trial) = SoS_skip_flag;
                    break;
%                 elseif keyCode(left_key) %change choice on previous trial
%                     SoS_choice(SoS_trial) = SoS_skip_flag;
%                     SoS_choice(SoS_trial-1) = -SoS_choice(SoS_trial-1);
%                     break;
                elseif keyCode(enter_key) %show instructions
                    trial_str = ['SoS_trial = ',num2str(SoS_trial)];
                    Screen('DrawText',SoS_w0, trial_str, SoS_x_1_2, SoS_y_1_4, blackCLUTndx);
                    [nx, ny, textbounds] = ...
                        DrawFormattedText(SoS_w0, instr_text, 'center', 'center', blackCLUTndx, SoS_x_3_4 );
                    Screen('Flip', SoS_w0);
                elseif keyCode(escape_key) %pause
                    trial_str = ['SoS_trial = ',num2str(SoS_trial)];
                    Screen('DrawText',SoS_w0, trial_str, SoS_x_1_2, SoS_y_1_4, blackCLUTndx);
                    Screen('Flip', SoS_w0);
                    WaitSecs(1.001) % set pause time
                    Screen('CloseAll');
                    WaitSecs(10.001) % set pause time
                    [w0, w0_rect] = Screen('OpenWindow',0);
                    old_text_size = Screen('TextSize', SoS_w0, default_text_size);
                    old_text_font = Screen('TextFont', SoS_w0);
                    old_text_style = Screen('TextStyle', SoS_w0);
                    Screen('DrawText',SoS_w0, trial_str, SoS_x_1_2, SoS_y_1_4, blackCLUTndx);
                    Screen('DrawText',SoS_w0, '''enter'' to continue, ''home'' to save, ''end'' to end', ...
                        SoS_x_1_4, SoS_y_3_4, blackCLUTndx);
                    Screen('Flip', w0);
                    while KbCheck; end  %clear keyboard buffer
                    while 1
                        [ keyIsDown, seconds, keyCode ] = KbCheck;
                        WaitSecs(0.001) % delay to prevent CPU hogging
                        if keyIsDown
                            if keyCode(enter_key) %resume trial
                                break;
                            elseif keyCode(end_key) %exit experiment
                                exit_experiment = 1;
                                break;
                            elseif keyCode(home_key) %save data
                                SoS_data.SoS_duration = SoS_duration(1:SoS_trial);
                                SoS_data.SoS_ISI = SoS_ISI(1:SoS_trial);
                                SoS_data.SoS_radius_x = SoS_radius_x(1:SoS_trial);
                                SoS_data.SoS_radius_y = SoS_radius_y(1:SoS_trial);
                                SoS_data.SoS_offset_x = SoS_offset_x(1:SoS_trial);
                                SoS_data.SoS_offset_y = SoS_offset_y(1:SoS_trial);
                                SoS_data.SoS_target = SoS_target(1:SoS_trial);
                                SoS_data.SoS_clutter = SoS_clutter(1:SoS_trial);
                                SoS_data.SoS_choice = SoS_choice(1:SoS_trial);
                                SoS_data.SoS_notarget = SoS_notarget(1:SoS_trial);
                                cd ('exp_data');
                                save(SoS_file_new, 'SoS_data');
                                cd ..
                                break;
                            else
                                continue;
                            end % if keyCode
                        end % if keyIsDown
                    end % while 1
                    while KbCheck; end  %clear keyboard buffer
                    Screen('FillRect',SoS_w0,grayCLUTndx);
                    Screen('Flip', SoS_w0);
                elseif keyCode(end_key) %punt
                    exit_experiment = 1;
                    break;
                else
                    continue;
                end % if keyCode
            end % if keyIsDown
            if exit_experiment == 1
                break;
            end % exit_experiment
            if mod(SoS_trial,100) == 0
                SoS_data.SoS_duration = SoS_duration(1:SoS_trial);
                SoS_data.SoS_radius_x = SoS_radius_x(1:SoS_trial);
                SoS_data.SoS_radius_y = SoS_radius_y(1:SoS_trial);
                SoS_data.SoS_offset_x = SoS_offset_x(1:SoS_trial);
                SoS_data.SoS_offset_y = SoS_offset_y(1:SoS_trial);
                SoS_data.SoS_target = SoS_target(1:SoS_trial);
                SoS_data.SoS_clutter = SoS_clutter(1:SoS_trial);
                SoS_data.SoS_choice = SoS_choice(1:SoS_trial);
                SoS_data.SoS_notarget = SoS_notarget(1:SoS_trial);
                cd ('exp_data');
                save(SoS_file_new, 'SoS_data');
                cd ..
            end
        end % while 1
        if exit_experiment == 1
            break;
        end % exit_experiment
    end % i_trial

    %% preprocess results
    SoS_duration = SoS_duration(1:SoS_trial);
    SoS_radius_x = SoS_radius_x(1:SoS_trial);
    SoS_radius_y = SoS_radius_y(1:SoS_trial);
    SoS_offset_x = SoS_offset_x(1:SoS_trial);
    SoS_offset_y = SoS_offset_y(1:SoS_trial);
    SoS_target = SoS_target(1:SoS_trial);
    SoS_clutter = SoS_clutter(1:SoS_trial);
    SoS_choice = SoS_choice(1:SoS_trial);
    SoS_notarget = SoS_notarget(1:SoS_trial);

    SoS_data.SoS_duration = SoS_duration(1:SoS_trial);
    SoS_data.SoS_radius_x = SoS_radius_x(1:SoS_trial);
    SoS_data.SoS_radius_y = SoS_radius_y(1:SoS_trial);
    SoS_data.SoS_offset_x = SoS_offset_x(1:SoS_trial);
    SoS_data.SoS_offset_y = SoS_offset_y(1:SoS_trial);
    SoS_data.SoS_target = SoS_target(1:SoS_trial);
    SoS_data.SoS_clutter = SoS_clutter(1:SoS_trial);
    SoS_data.SoS_choice = SoS_choice(1:SoS_trial);
    SoS_data.SoS_notarget = SoS_notarget(1:SoS_trial);

    Screen('Close',SoS_w0)
catch
    Screen('CloseAll')
    rethrow(lasterror)
end

%% save and append to new version of existing file
cd ('exp_data');
save(SoS_file_new, 'SoS_data');
SoS_data_new = SoS_data;
SoS_file = SoS_file_new; %default
SoS_dialog_append = questdlg('append data to existing file?', 'save data', 'No');
if strcmp(SoS_dialog_append, 'Yes')
    exp_data_dir = dir;
    exp_data_datenum = zeros( length(exp_data_dir), 1 );
    for i_file = 1 : length(exp_data_dir)
        exp_data_datenum(i_file) = datenum(exp_data_dir(i_file).date);
    end
    [exp_data_datenum_sorted, exp_data_ndx_sorted] = sort(exp_data_datenum, 'descend');
    exp_data_dir = exp_data_dir(exp_data_ndx_sorted);
    exp_data_files = {exp_data_dir.name};
    exp_data_default_ndx = 1;
    exp_data_default_date = exp_data_dir(exp_data_default_ndx).date;
    [exp_data_file_ndx, exp_data_file_flag] = ...
        listdlg('PromptString','Select a file:', ...
        'SelectionMode','single', ...
        'InitialValue', exp_data_default_ndx, ...
        'ListString',{exp_data_dir.name});
    if exp_data_file_flag ~= 0
        SoS_file_old = exp_data_dir(exp_data_file_ndx).name;
        if ~strcmp(SoS_file_old,SoS_file_new)
            load(SoS_file_old, 'SoS_data');
            SoS_data.SoS_files = [SoS_data.SoS_files; {SoS_file_new}];
            SoS_data.SoS_duration = [SoS_data.SoS_duration; SoS_data_new.SoS_duration];
            SoS_data.SoS_radius_x = [SoS_data.SoS_radius_x; SoS_data_new.SoS_radius_x];
            SoS_data.SoS_radius_y = [SoS_data.SoS_radius_y; SoS_data_new.SoS_radius_y];
            SoS_data.SoS_offset_x = [SoS_data.SoS_offset_x; SoS_data_new.SoS_offset_x];
            SoS_data.SoS_offset_y = [SoS_data.SoS_offset_y; SoS_data_new.SoS_offset_y];
            SoS_data.SoS_target = [SoS_data.SoS_target; SoS_data_new.SoS_target];
            SoS_data.SoS_clutter = [SoS_data.SoS_clutter_vals; SoS_data_new.SoS_clutter];
            SoS_data.SoS_choice = [SoS_data.SoS_choice; SoS_data_new.SoS_choice];
            SoS_data.SoS_notarget = [SoS_data.SoS_notarget; SoS_data_new.SoS_notarget];
            char_tmp = SoS_file_old(end-7);
            if strcmp(char_tmp,'A')
                append_ver = (SoS_file_old(end-6:end-4));
                append_ver = mod(str2double(append_ver) + 1, 999);
                append_format = '%03u';
                append_str = num2str(append_ver, append_format);
                SoS_file_append = [ SoS_file_old(1:end-7), append_str, '.mat'];
            else
                SoS_file_append = [ SoS_file_old(1:end-4), 'A001.mat'];
            end
            save(SoS_file_append, 'SoS_data');
        end
    end
end
cd ..



%% analyze data
SoS_duration_delta = SoS_data.SoS_duration_delta; %bin width for analysis
SoS_target_delta = SoS_data.SoS_target_delta; %bin width for analysis
SoS_clutter_delta = SoS_data.SoS_clutter_delta; %bin width for analysis;
SoS_duration_vals = SoS_data.SoS_duration_vals;
SoS_target_vals = SoS_data.SoS_target_vals;
SoS_clutter_vals = SoS_data.SoS_clutter_vals;
SoS_edges_duration = ...
    [ ( SoS_duration_vals - ( SoS_duration_delta / 2 ) ), ...
    ( SoS_duration_max + ( SoS_duration_delta / 2 ) ) ];
SoS_edges_target = ...
    [ ( SoS_target_vals - ( SoS_target_delta / 2 ) ), ...
    ( SoS_target_max + ( SoS_target_delta / 2 ) ) ];
SoS_edges_clutter = ...
    [ ( SoS_clutter_vals - ( SoS_clutter_delta / 2 ) ), ...
    ( SoS_clutter_max + ( SoS_clutter_delta / 2 ) ) ];

valid_ndx = find(SoS_data.SoS_choice <= 1 & SoS_data.SoS_choice >= -1);
first_valid_ndx = find( valid_ndx > 0, 1, 'first' );
valid_ndx = valid_ndx(first_valid_ndx:end);
valid_trials = length(valid_ndx);

SoS_duration2 = SoS_data.SoS_duration(valid_ndx);
SoS_target2 = SoS_data.SoS_target(valid_ndx);
SoS_clutter2 = SoS_data.SoS_clutter(valid_ndx);
SoS_choice2 = SoS_data.SoS_choice(valid_ndx);
SoS_notarget2 = SoS_data.SoS_notarget(valid_ndx);

correct_ndx = ...
    find( ( ( SoS_notarget2 == 1 ) & ( SoS_choice2 == 0 ) ) | ...
    ( ( SoS_notarget2 == 0 ) & ( SoS_choice2 == 1 ) ) );
incorrect_ndx = ...
    find( ( ( SoS_notarget2 == 0 ) & ( SoS_choice2 == 0 ) ) | ...
    ( ( SoS_notarget2 == 1 ) & ( SoS_choice2 == 1 ) ) );

num_correct = zeros( length(SoS_edges_duration), length(SoS_edges_clutter) );
SoS_Webber = 0;
if length(correct_ndx) > 0
    num_correct = hist3( [ SoS_duration2(correct_ndx), SoS_clutter2(correct_ndx) ], 'Edges', ...
        {SoS_edges_duration', SoS_edges_clutter'} );
    num_correct = ...
        num_correct(1:end - 1, 1:end - 1);
end
if length(incorrect_ndx) > 0
    num_incorrect = hist3( [ SoS_duration2(incorrect_ndx), SoS_clutter2(incorrect_ndx) ], 'Edges', ...
        {SoS_edges_duration', SoS_edges_clutter'} );
    num_incorrect = ...
        num_incorrect(1:end - 1, 1:end - 1);
end
num_tot = num_correct + num_incorrect;
percent_correct = num_correct ./ ( num_tot + (num_tot == 0) );


figure
lh = plot( percent_correct );




