clear all
%echo on
exit_experiment = 0;
SoS_seed = sum(100*clock);
rand('state', SoS_seed); % reseed the random-number generator for each expt.
SoS_state = rand('state');
SoS_src_path = 'C:\cygwin\home\gkenyon\trunk\psycho\';
SoS_data_path = [SoS_src_path, 'exp_data'];
cd (SoS_src_path);
if ~exist(SoS_data_path,'dir')
    SoS_dialog_mkdir = questdlg('make exp_data directory?', 'make directory', 'Yes');
    if strcmp(SoS_dialog_mkdir, 'Yes')
        mkdir(SoS_data_path);
    elseif strcmp(SoS_dialog_mkdir, 'No') || strcmp(SoS_dialog_mkdir, 'Cancel')
        exit_experiment = 1;
    end
end

if exit_experiment == 0
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
        % use KbName('KeyNames') to define equivalents for different operating
        % systems
        escape_key = KbName('esc'); %on mac OSX, 'Escape'
        up_key = KbName('up'); %on mac OSX, 'UpArrow'
        down_key = KbName('down'); %on mac OSX, 'DownArrow'
        left_key = KbName('left'); %on mac OSX, 'LeftArrow'
        right_key = KbName('right'); %on mac OSX, 'RightArrow'
        enter_key = KbName('return'); %on mac OSX, 'ENTER'
        end_key = KbName('end'); %on mac OSX, 'End'
        home_key = KbName('home'); %on mac OSX, 'Home'


        %% init target and clutter params
        SoS_num_trials = 10; %number of trials for each condition
        SoS_num_duration_vals = 3; %number of durations per trial
        SoS_duration_min = 0.5 * frame_duration; %min/max viewing duration
        SoS_duration_max = 2.5 * frame_duration;
        SoS_duration_delta = ( SoS_duration_max - SoS_duration_min + ( SoS_num_duration_vals <= 1 ) ) / ...
            ( SoS_num_duration_vals - 1 + ( SoS_num_duration_vals <= 1 ) ); %bin width for analysis
        SoS_duration_vals = ...
            SoS_duration_min : SoS_duration_delta : SoS_duration_max;
        SoS_IMAGE_FROM_DATABASE = 0;
        SoS_IMAGE_FROM_RENDER = 1;
        SoS_image_source = SoS_IMAGE_FROM_DATABASE;
        SoS_grayscale_flag = 1; % 0 to use color (if available)
        SoS_resize_flag = 0; % 0 to use original image size, 1 rescales foreground rect
        SoS_file_new = ['SoSExp', num2str(round(SoS_seed)), '.mat'];
        SoS_original_rect = SoS_foreground_rect;

        if SoS_image_source == SoS_IMAGE_FROM_DATABASE
            SoS_DB_path = 'C:\cygwin\home\gkenyon\AnimalDB\';
            SoS_num_target_labels = 4;
            SoS_target_files = cell(SoS_num_target_labels, 1);
            SoS_num_target_images = zeros(SoS_num_target_labels, 1);
            SoS_target_image_ndx = cell(SoS_num_target_labels, 1);
            SoS_target_counter = zeros(SoS_num_target_labels, 1);
            SoS_target_folder = [SoS_DB_path, 'Targets\'];
            SoS_target_prefix = {'H_', 'B_', 'M_', 'F_'};
            for i_label = 1:SoS_num_target_labels
                str_tmp = [SoS_target_folder, SoS_target_prefix{i_label}, '*'];
                SoS_target_files{i_label} = dir(str_tmp);
                SoS_num_target_images(i_label) = length( SoS_target_files{i_label} );
                [tmp, SoS_target_image_ndx{i_label}] = ...
                    Shuffle(rand(SoS_num_target_images(i_label),1));
            end
            SoS_num_control_labels = 4;
            SoS_control_files = cell(SoS_num_control_labels, 1);
            SoS_num_control_images = zeros(SoS_num_control_labels, 1);
            SoS_control_image_ndx = cell(SoS_num_control_labels, 1);
            SoS_control_counter = zeros(SoS_num_control_labels, 1);
            SoS_control_folder = [SoS_DB_path, 'Distractors\'];
            SoS_control_prefix = {'Hdn_', 'Bdn_', 'Mdn_', 'Fdn_'}; % {''};%
            for i_label = 1:SoS_num_control_labels
                str_tmp = [SoS_control_folder, SoS_control_prefix{i_label}, '*.jpg'];
                SoS_control_files{i_label} = dir(str_tmp);
                SoS_num_control_images(i_label) = length( SoS_control_files{i_label} );
                [tmp, SoS_control_image_ndx{i_label}] = shuffle(rand(SoS_num_control_images(i_label),1));
            end
        elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
            SoS_num_clutter_vals = 2;
            SoS_clutter_min = 2000;  %min/max percent clutter
            SoS_clutter_max = 5000;
            SoS_clutter_delta = (SoS_clutter_max - SoS_clutter_min + ( SoS_num_clutter_vals <= 1 ) ) / ...
                ( SoS_num_clutter_vals - 1 + ( SoS_num_clutter_vals <= 1 ) ); %bin width for analysis
            SoS_clutter_vals = ...
                SoS_clutter_min : SoS_clutter_delta : SoS_clutter_max;
            SoS_num_target_labels = SoS_num_clutter_vals;
            SoS_num_control_labels = SoS_num_clutter_vals;
            SoS_radius_min = 0.1; %min/max target radius (fraction of foreground rectangle)
            SoS_radius_max = 0.2;
            SoS_offset_min = -0.1; %min/max target center (fraction of foreground rectangle)
            SoS_offset_max = 0.1;
        end

        SoS_files = {SoS_file_new};
        SoS_ITI = 1; %inter-trial-interval
        SoS_gray_mid = mean(grayCLUTndx);
        SoS_gray_min = blackCLUTndx;%
        SoS_gray_max = whiteCLUTndx;%
        SoS_skip_flag = 10; %trial skipped
        SoS_tot_combinations = 2 * SoS_num_duration_vals * SoS_num_target_labels;
        SoS_tot_trials = SoS_num_trials * SoS_tot_combinations;
        SoS_duration = zeros(SoS_tot_trials,1);
        SoS_target_label = zeros(SoS_tot_trials,1);
        SoS_control_label = zeros(SoS_tot_trials,1);
        SoS_choice = repmat( SoS_skip_flag , SoS_tot_trials, 1 );
        SoS_target_flag = ones(SoS_tot_trials,1);
        if SoS_image_source == SoS_IMAGE_FROM_DATABASE
            SoS_DB_file = cell(SoS_tot_trials,1);
        elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
            SoS_radius_x = zeros(SoS_tot_trials,1);
            SoS_radius_y = zeros(SoS_tot_trials,1);
            SoS_offset_x = zeros(SoS_tot_trials,1);
            SoS_offset_y = zeros(SoS_tot_trials,1);
        end
        SoS_trial_index = repmat( 1 : SoS_tot_combinations, SoS_num_trials, 1 );
        SoS_trial_index = reshape( SoS_trial_index, SoS_tot_trials, 1 );
        [SoS_trial_index, SoS_shuffle_index] = Shuffle(SoS_trial_index);

        VBLTimestamp = zeros(SoS_tot_trials,2);
        StimulusOnsetTime = zeros(SoS_tot_trials,2);
        FlipTimestamp = zeros(SoS_tot_trials,2);
        Missed = zeros(SoS_tot_trials,2);
        Beampos = zeros(SoS_tot_trials,2);

        SoS_data = struct( ...
            'SoS_seed', SoS_seed, ...
            'SoS_state', SoS_state, ...
            'SoS_ITI', SoS_ITI, ...
            'SoS_duration_min', SoS_duration_min, ...
            'SoS_duration_max', SoS_duration_max, ...
            'SoS_duration', SoS_duration, ...
            'SoS_num_duration_vals', SoS_num_duration_vals, ...
            'SoS_duration_delta', SoS_duration_delta, ...
            'SoS_duration_vals', SoS_duration_vals, ...
            'SoS_num_target_labels', SoS_num_target_labels, ...
            'SoS_num_control_labels', SoS_num_control_labels, ...
            'SoS_target_label', SoS_target_label, ...
            'SoS_control_label', SoS_control_label, ...
            'SoS_gray_mid', SoS_gray_mid, ...
            'SoS_gray_min', SoS_gray_min, ...
            'SoS_gray_max', SoS_gray_max, ...
            'SoS_skip_flag', SoS_skip_flag, ...
            'SoS_choice', SoS_choice, ...
            'SoS_target_flag', SoS_target_flag, ...
            'SoS_grayscale_flag', SoS_grayscale_flag, ...
            'SoS_resize_flag', SoS_resize_flag, ...
            'VBLTimestamp', VBLTimestamp, ...
            'StimulusOnsetTime', StimulusOnsetTime, ...
            'FlipTimestamp', FlipTimestamp, ...
            'Missed', Missed, ...
            'Beampos', Beampos, ...
            'SoS_files', SoS_files, ...
            'SoS_image_source', SoS_image_source ...
            );
        if SoS_image_source == SoS_IMAGE_FROM_DATABASE
            SoS_data.SoS_target_files = SoS_target_files;
            SoS_data.SoS_control_files = SoS_control_files;
            SoS_data.SoS_DB_file = SoS_DB_file;
        elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
            SoS_data.SoS_radius_min = SoS_radius_min;
            SoS_data.SoS_radius_max = SoS_radius_max;
            SoS_data.SoS_offset_min = SoS_offset_min;
            SoS_data.SoS_offset_max = SoS_offset_max;
            SoS_data.SoS_radius_x = SoS_radius_x;
            SoS_data.SoS_radius_y = SoS_radius_y;
            SoS_data.SoS_offset_x = SoS_offset_x;
            SoS_data.SoS_offset_y = SoS_offset_y;
        end


        %set background color
        Screen('FillRect', SoS_w0, SoS_gray_mid);
        Screen('Flip', SoS_w0);

        %% start experiment
        SoS_correct_tally = 0;
        SoS_incorrect_tally = 0;
        SoS_edge_len = 16*sqrt(2);
        SoS_penWidth = 3;
        for SoS_trial = 1 : SoS_tot_trials
            [SoS_target_flag(SoS_trial), SoS_duration(SoS_trial), SoS_target_label(SoS_trial)] = ...
                ind2sub([2, SoS_num_duration_vals, SoS_num_target_labels], ...
                SoS_trial_index(SoS_trial));
            SoS_duration(SoS_trial) = SoS_duration_vals(SoS_duration(SoS_trial));
            SoS_target_flag(SoS_trial) = SoS_target_flag(SoS_trial) - 1;
            Screen('DrawText',SoS_w0, 'drawing image...', SoS_x_1_2, SoS_y_1_2, blackCLUTndx);
            Screen('Flip', SoS_w0);

            if SoS_image_source == SoS_IMAGE_FROM_DATABASE
                if SoS_num_control_labels == SoS_num_target_labels
                    SoS_control_label(SoS_trial) = SoS_target_label(SoS_trial);
                else
                    SoS_control_label(SoS_trial) = 1 + fix( SoS_num_control_labels * rand(1) );
                end
                if SoS_target_flag(SoS_trial) == 1
                    SoS_target_counter(SoS_target_label(SoS_trial)) = ...
                        SoS_target_counter(SoS_target_label(SoS_trial)) + 1;
                    if SoS_target_counter(SoS_target_label(SoS_trial)) > ...
                            SoS_num_target_images(SoS_target_label(SoS_trial))
                        [tmp, SoS_target_image_ndx{SoS_target_label(SoS_trial)}] = ...
                            shuffle(rand(SoS_num_target_images(SoS_target_label(SoS_trial)),1));
                        SoS_target_counter(SoS_target_label(SoS_trial)) = 0;
                    end
                    SoS_DB_file{SoS_trial} =  ...
                        SoS_target_files{ SoS_target_label(SoS_trial) } ...
                        ( SoS_target_image_ndx{ SoS_target_label(SoS_trial) }...
                        ( SoS_target_counter( SoS_target_label(SoS_trial) ) ) );
                    SoS_DB_file{SoS_trial} =  ...
                        [ SoS_target_folder, SoS_DB_file{SoS_trial}.name ];
                    SoS_image = imread( SoS_DB_file{SoS_trial} );
                else
                    SoS_control_counter(SoS_control_label(SoS_trial)) = ...
                        SoS_control_counter(SoS_control_label(SoS_trial)) + 1;
                    if SoS_control_counter(SoS_control_label(SoS_trial)) > ...
                            SoS_num_control_images(SoS_control_label(SoS_trial))
                        [tmp, SoS_control_image_ndx{i_label}] = ...
                            shuffle(rand(SoS_num_control_images(SoS_control_label(SoS_trial)),1));
                        SoS_target_counter(SoS_target_label(SoS_trial)) = 0;
                    end
                    SoS_DB_file{SoS_trial} =  ...
                        SoS_control_files{ SoS_control_label(SoS_trial) } ...
                        ( SoS_control_image_ndx{ SoS_control_label(SoS_trial) }...
                        ( SoS_control_counter( SoS_control_label(SoS_trial) ) ) );
                    SoS_DB_file{SoS_trial} =  ...
                        [ SoS_control_folder, SoS_DB_file{SoS_trial}.name ];
                    SoS_image = imread( SoS_DB_file{SoS_trial} );
                end
                if SoS_grayscale_flag
                    SoS_image = .2989*SoS_image(:,:,1)...
                        +.5870*SoS_image(:,:,2)...
                        +.1140*SoS_image(:,:,3);
                end
                SoS_noise_image = Shuffle(SoS_image(:));
                SoS_noise_image = reshape( SoS_noise_image, size(SoS_image) );
                SoS_original_rect = ...
                    [0 0 ( size(SoS_image,1)-1 ) ( size(SoS_image,2)-1 )  ];
                SoS_image_rect = ...
                    [( SoS_x_1_2-size(SoS_image,1)+1 ) ...
                    ( SoS_y_1_2-size(SoS_image,2)+1 ) ...
                    ( SoS_x_1_2+size(SoS_image,1) )...
                    ( SoS_y_1_2+size(SoS_image,2) )  ];
                if SoS_resize_flag
                    Screen( 'PutImage', SoS_w0, SoS_image, SoS_foreground_rect );
                else
                   Screen( 'PutImage', SoS_w0, SoS_image, SoS_image_rect );
                end
            elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
                if SoS_target_flag(SoS_trial) == 1
                    SoS_radius_x(SoS_trial) = ...
                        fix( ( SoS_radius_min + rand * (SoS_radius_max - SoS_radius_min) ) * ...
                        SoS_foreground_width );
                    SoS_radius_y(SoS_trial) = ...
                        fix( ( SoS_radius_min + rand * (SoS_radius_max - SoS_radius_min) ) * ...
                        SoS_foreground_height );
                    SoS_offset_x(SoS_trial) = ...
                        fix( ( SoS_offset_min + rand * (SoS_offset_max - SoS_offset_min) ) * ...
                        SoS_foreground_width );
                    SoS_offset_y(SoS_trial) = ...
                        fix( ( SoS_offset_min + rand * (SoS_offset_max - SoS_offset_min) ) * ...
                        SoS_foreground_height );
                    SoS_target_rect_left = ...
                        SoS_x_1_2 - SoS_radius_x(SoS_trial) + SoS_offset_x(SoS_trial);
                    SoS_target_rect_top = ...
                        SoS_y_1_2 + SoS_radius_y(SoS_trial) + SoS_offset_y(SoS_trial);
                    SoS_target_rect_right = ...
                        SoS_x_1_2 + SoS_radius_x(SoS_trial) + SoS_offset_x(SoS_trial);
                    SoS_target_rect_bottom = ...
                        SoS_y_1_2 - SoS_radius_y(SoS_trial) + SoS_offset_y(SoS_trial);
                    SoS_target_rect = ...
                        [SoS_target_rect_left SoS_target_rect_bottom ...
                        SoS_target_rect_right SoS_target_rect_top];
                    Screen( 'FrameOval', SoS_w0, SoS_gray_max, SoS_target_rect, SoS_penWidth );
                end % SoS_target_flag
                % make clutter
                SoS_control_label(SoS_trial) = SoS_clutter_vals(SoS_target_label(SoS_trial));
                SoS_num_clutter = SoS_control_label(SoS_trial);
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
                Screen('DrawLines', SoS_w0, SoS_clutter_lines, SoS_penWidth, SoS_gray_max, ...
                    [SoS_x_1_2, SoS_y_1_2]);
                SoS_image = Screen('GetImage', SoS_w0, SoS_foreground_rect, 'backBuffer');
                SoS_image = .2989*SoS_image(:,:,1)...
                    +.5870*SoS_image(:,:,2)...
                    +.1140*SoS_image(:,:,3);
                SoS_noise_image = Shuffle(SoS_image(:));
                SoS_noise_image = reshape( SoS_noise_image, size(SoS_image) );
                SoS_image_rect = SoS_foreground_rect;
            end

            if SoS_resize_flag
                SoS_noise_rect = SoS_foreground_rect;
            else
                SoS_noise_rect = SoS_image_rect;
            end
            SoS_noise_texture = Screen('MakeTexture', SoS_w0, SoS_noise_image);

            % draw image
            max_priority = MaxPriority(SoS_w0, 'WaitSecs');
            SoS_stim = {
                'WaitSecs(SoS_ITI);'
                '[VBLTimestamp(SoS_trial,1) StimulusOnsetTime(SoS_trial,1) FlipTimestamp(SoS_trial,1) Missed(SoS_trial,1) Beampos(SoS_trial,1)] = Screen(''Flip'', SoS_w0);'
                'WaitSecs(SoS_duration(SoS_trial));'
                'Screen(''DrawTexture'', SoS_w0, SoS_noise_texture, SoS_original_rect, SoS_noise_rect);'
                '[VBLTimestamp(SoS_trial,2) StimulusOnsetTime(SoS_trial,2) FlipTimestamp(SoS_trial,2) Missed(SoS_trial,2) Beampos(SoS_trial,2)] = Screen(''Flip'', SoS_w0);'
                };
            %Screen('DrawTexture', windowPointer, texturePointer [,sourceRect] [,destinationRect] [,rotationAngle] [, filterMode] [, globalAlpha]);
            
            Rush( SoS_stim, max_priority );

            %parse user keybord input (note some options are nested to allow for pausing experiment)
            while KbCheck; end  %clears keyboard buffer from previous trial
            while 1
                [ keyIsDown, seconds, keyCode ] = KbCheck;
                WaitSecs(0.001) % delay to prevent CPU hogging
                if keyIsDown
                    if keyCode(up_key) % taget present
                        SoS_choice(SoS_trial) = 1;
                        if SoS_target_flag(SoS_trial) == 0
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
                        if SoS_target_flag(SoS_trial) == 1
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
                                    SoS_SaveData;
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
                    SoS_SaveData;
                end
            end % while 1
            if exit_experiment == 1
                break;
            end % exit_experiment
        end % i_trial

        %% preprocess results
        SoS_actual_duration = cell(3,1);
        SoS_actual_duration{1,1} = StimulusOnsetTime(1:SoS_trial,2) - StimulusOnsetTime(1:SoS_trial,1);
        SoS_actual_duration{2,1} = VBLTimestamp(1:SoS_trial,2) - VBLTimestamp(1:SoS_trial,1);
        SoS_actual_duration{3,1} = FlipTimestamp(1:SoS_trial,2) - FlipTimestamp(1:SoS_trial,1);
        SoS_data.SoS_actual_duration = SoS_actual_duration;
        SoS_SaveData
        Screen('Close',SoS_w0)
    catch
        Screen('CloseAll')
        rethrow(lasterror)
    end
end


%% save and append to new version of existing file
cd (SoS_data_path);
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
            SoS_data.SoS_target_label = [SoS_data.SoS_target_label; SoS_data_new.SoS_target_label];
            SoS_data.SoS_control_label = [SoS_data.SoS_control_label; SoS_data_new.SoS_control_label];
            SoS_data.SoS_choice = [SoS_data.SoS_choice; SoS_data_new.SoS_choice];
            SoS_data.SoS_target_flag = [SoS_data.SoS_target_flag; SoS_data_new.SoS_target_flag];
            if SoS_image_source == SoS_IMAGE_FROM_DATABASE
                SoS_data.SoS_DB_file = [SoS_data.SoS_DB_file; SoS_data_new.SoS_DB_file];
            elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
                SoS_data.SoS_radius_x = [SoS_data.SoS_radius_x; SoS_data_new.SoS_radius_x];
                SoS_data.SoS_radius_y = [SoS_data.SoS_radius_y; SoS_data_new.SoS_radius_y];
                SoS_data.SoS_offset_x = [SoS_data.SoS_offset_x; SoS_data_new.SoS_offset_x];
                SoS_data.SoS_offset_y = [SoS_data.SoS_offset_y; SoS_data_new.SoS_offset_y];
            end
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
cd(SoS_src_path);



%% analyze data
SoS_duration_delta = SoS_data.SoS_duration_delta; %bin width for analysis
SoS_duration_vals = SoS_data.SoS_duration_vals;
SoS_target_label = SoS_data.SoS_target_label;
SoS_control_label = SoS_data.SoS_control_label;
SoS_edges_duration = ...
    [ ( SoS_duration_vals - ( SoS_duration_delta / 2 ) ), ...
    ( SoS_duration_max + ( SoS_duration_delta / 2 ) ) ];

valid_ndx = find(SoS_data.SoS_choice <= 1 & SoS_data.SoS_choice >= -1);
first_valid_ndx = find( valid_ndx > 0, 1, 'first' );
valid_ndx = valid_ndx(first_valid_ndx:end);
valid_trials = length(valid_ndx);

SoS_duration2 = SoS_data.SoS_duration(valid_ndx);
SoS_target_label2 = SoS_data.SoS_target_label(valid_ndx);
SoS_control_label2 = SoS_data.SoS_control_label(valid_ndx);
SoS_choice2 = SoS_data.SoS_choice(valid_ndx);
SoS_target_flag2 = SoS_data.SoS_target_flag(valid_ndx);

correct_ndx = ...
    find( ( ( SoS_target_flag2 == 0 ) & ( SoS_choice2 == 0 ) ) | ...
    ( ( SoS_target_flag2 == 1 ) & ( SoS_choice2 == 1 ) ) );
incorrect_ndx = ...
    find( ( ( SoS_target_flag2 == 0 ) & ( SoS_choice2 == 1 ) ) | ...
    ( ( SoS_target_flag2 == 1 ) & ( SoS_choice2 == 0 ) ) );

num_correct = zeros( SoS_num_duration_vals, SoS_num_target_labels );
SoS_Webber = 0;
if length(correct_ndx) > 0
    num_correct = hist3( [ SoS_duration2(correct_ndx), SoS_target_label2(correct_ndx) ], ...
        { SoS_duration_vals, 1:SoS_num_target_labels} );
    if length(incorrect_ndx) > 0
        num_incorrect = hist3( [ SoS_duration2(incorrect_ndx), SoS_target_label2(incorrect_ndx) ], ...
            { SoS_duration_vals, 1:SoS_num_target_labels} );
        num_tot = num_correct + num_incorrect;
        percent_correct = num_correct ./ ( num_tot + (num_tot == 0) );
        figure
        lh = plot( (0.5 * SoS_duration_delta+SoS_duration_vals), percent_correct );
        set(lh, 'LineWidth', 2.0);
        SoS_markers = ['o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>'];
        for i_label = 1:SoS_num_target_labels
            set(lh(i_label), 'Marker', SoS_markers(i_label));
        end
        if SoS_image_source == SoS_IMAGE_FROM_DATABASE
            leg_h = legend('head', 'body', ' mid', ' far');
            set(leg_h, 'Location', 'SouthEast');
        elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
        end
    end
end


