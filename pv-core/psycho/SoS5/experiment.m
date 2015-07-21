function experiment()
    global w0 screen_xC screen_yC w h ver refresh_rate manual_timing exit_experiment exp

    r = 0.5*refresh_rate;
    fix_xy = [screen_xC, screen_xC, screen_xC+50, screen_xC-50; screen_yC+50, screen_yC-50, screen_yC, screen_yC];      %fixation coordinates
    %exp.isi = 1;  %inter stimulus interval, in frames
    exp.fixation_interval = .40;    %interval that the fixation object is present  
    
    for trial = 1 : exp.tot_trials
        %% fixation object
        %Screen('DrawText',w0, 'drawing image...', w0_rect(3)/2-120, w0_rect(4)/2, BlackIndex(w0));
        %Screen('FillOval',w0,255,[screen_xC-15 screen_yC-15 screen_xC+15 screen_yC+15]);
        Screen('DrawLines',w0, fix_xy, 3);
        exp.t4(trial) = GetSecs;
        f0 = Screen('Flip', w0);
        exp.t5(trial) = GetSecs;
    
        exp.duration(trial) = Sample(exp.duration_vals);

        if ~exp.amoeba

            image = imread(exp.shuffled_files(trial).name, 'jpeg');
          
            if exp.grayscale_flag
                image = .2989*image(:,:,1) ...
                    +.5870*image(:,:,2) ...
                    +.1140*image(:,:,3);
            end

            if strcmp(exp.shuffled_files(trial).name(2),'d')
                exp.target_flag = 0;
            end

            im_w = size(image,1)-1;
            im_h = size(image,2)-1;
            
            image_rect = [( screen_xC-im_w ) ( screen_yC-im_h ) ( screen_xC+im_w ) ( screen_yC+im_h )];
            
            original_rect = ...
                [0 0 ( im_w ) ( im_h )  ];
           
            image_tex = Screen('MakeTexture',w0 ,image);
            Screen('DrawTexture', w0, image_tex, original_rect, image_rect);
     
        else
            if ver(3) == '6'
                
                if exp.target_flag(trial)
                    target = Segment('target');
                    distractor = Segment.empty(exp.segment_total - target.num_t, 0);
                    for i = 1:(exp.segment_total - target.num_t)
                        distractor(i) = Segment('distractor');
                    end
                    exp.obj{trial,1} = distractor;
                    exp.obj{trial,2} = target;
                else
                    distractor = Segment.empty(exp.segment_total,0);
                    for i=1:exp.segment_total
                        distractor(i) = Segment('distractor');
                    end
                    exp.obj{trial,1} = distractor;
                end
            else
                
                if exp.target_flag(trial)
                    exp.obj{trial,1} = segmentStruct('target');
                    exp.obj{trial,2} = segmentStruct('distractors',exp.obj{trial,1}.num_t);
                else
                    exp.obj{trial,1} = segmentStruct('no_target');
                end
            end
            image_rect = [screen_xC-w/2 screen_yC-h/2 screen_xC+w/2 screen_yC+h/2];
            image = Screen('GetImage', w0, image_rect, 'backBuffer');

            % apply the luminance equation
            % this is faster than only grabbing luminance from GetImage
            %  (maybe ?)
            if exp.grayscale_flag
                image = .2989*image(:,:,1) ...
                    +.5870*image(:,:,2) ...
                    +.1140*image(:,:,3);
            end

        end


        if exp.mask_mode == 0
            noise_image = Shuffle(image(:));
            noise_image = reshape( noise_image , size(image) );
            noise_tex = Screen('MakeTexture', w0, noise_image);
        end

        original_rect = ...
            [0 0 ( size(image,1)-1 ) ( size(image,2)-1 )  ];
        
       %mask(exp.obj{trial,:});
       %if exp.target_flag(trial); mask(target, distractor); else
       %mask(distractor); end;
       
       % draw image
        %max_priority = MaxPriority(w0, 'WaitSecs');
        max_priority = MaxPriority(w0);
        if manual_timing
            stim = {
                'exp.t0(trial) = GetSecs;'
                'f1 = Screen(''Flip'', w0, f0 + exp.fixation_interval);'
                'exp.t1(trial) = GetSecs;' %t1-t0 is an estimate of FlipTimestamp - VBLTimestamp
                % 'WaitSecs(.5);'
                % 'f2 = Screen(''Flip'', w0, f1 + exp.isi*r);'
                 'Screen(''DrawTexture'', w0, noise_tex, original_rect, image_rect );'
                % 'if exp.target_flag(trial); mask(exp.segment_total, distractor, target); else mask(exp.segment_total, distractor); end;' 
                'exp.t2(trial) = GetSecs;'
                'f3 = Screen(''Flip'', w0, f1 + r);'
                'exp.t3(trial) = GetSecs;' %t3-t1 is an estimation of the how much time elapsed between flips
                };

            %Screen('DrawTexture', windowPointer, texturePointer [,sourceRect] [,destinationRect] [,rotationAngle] [, filterMode] [, globalAlpha]);
        else
            stim = {
                %'WaitSecs(exp.ITI);'
                't0(trial) = GetSecs;'
                '[exp.VBLTimestamp(trial,1) exp.StimulusOnsetTime(trial,1) exp.FlipTimestamp(trial,1) exp.Missed(trial,1) exp.Beampos(trial,1)] = Screen(''Flip'', w0);'
                't1(trial) = GetSecs;' %t1-t0 is an estimation of FlipTimestamp - VBLTimestamp
                'WaitSecs(exp.duration(trial));'
                %'WaitSecs(3);'
               
                %'Screen(''DrawTexture'', w0, noise_tex, noise_rect);'
                %'t2(trial) = GetSecs;'
                '[exp.VBLTimestamp(trial,2) exp.StimulusOnsetTime(trial,2) exp.FlipTimestamp(trial,2) exp.Missed(trial,2) exp.Beampos(trial,2)] = Screen(''Flip'', w0);'
                't3(trial) = GetSecs;' %t3-t1 is an estimation of the how much time elapsed between flips
                };
            %Screen('DrawTexture', windowPointer, texturePointer
            %[,sourceRect] [,destinationRect] [,rotationAngle] [, filterMode] [, globalAlpha]);
        end
        Rush( stim, max_priority );
        s0 = GetSecs;
        [exp.choice(trial) exp.key_name{trial} sN sD] = keyPress(trial);
        exp.response_time(trial,:) = [(sN-s0) sD];

        if exp.training
            [exp.num_correct exp.num_incorrect exp.num_skipped] = score(trial, exp.choice, exp.target_flag);
            tally_str = ['% correct = ',num2str(exp.num_correct/(exp.num_correct + exp.num_incorrect))];
            Screen('DrawText',w0, tally_str, w0_rect(3)/2-100, w0_rect(4)/2, 0);
            Screen('Flip', w0);
            pause(.5);
        end

        if mod(trial,100) == 0
            save(data_file);
        end

        if exit_experiment
            ListenChar(0);
            Screen('CloseAll');
            break;
        end % exit_experiment

        if exp.save_mode
            save(data_file);
            fopen([data_file '_mask' '.jpg'], 'w+');
            imwrite(noise_image, [data_file '_mask' '.jpg']);
            if amoeba
                fopen([data_file '.tiff'], 'w+');
                imwrite(image, [data_file '.tiff']);
            end
        end
        Screen('Close');
    end
end

