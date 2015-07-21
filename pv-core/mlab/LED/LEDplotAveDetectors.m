function fh_detector = ...
    LEDplotAveDetectors( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, mask_str_list, ...
    data_SoA, num_SoAs_data, max_time, ...
    LED_target, LED_distractor_target, ...
    LED_target_mask, LED_distractor_mask, ...
    LED_distractor, LED_target_distractor)

global start_target stop_target
global start_distractor stop_distractor

num_distractors = stop_distractor - start_distractor + 1;
fh_detector = zeros(num_targets, 1, num_SoAs_data);
for i_target = start_target : stop_target
    for i_SoA = 2 : 2 : num_SoAs_data
        i_mask = 1;
        fh_detector(i_target, 1, i_SoA) = ...
            figure( 'Name', ...
            ['target (',   target_str_list{i_target}, ...
            ') vs. time; ', ...
            'SoA = ', num2str(data_SoA(i_SoA)) ] );
        subplot(num_masks, 2, 1);
        axis( [ 0 max_time -1.2 1.2 ] );
        hold on
        
        ave_LED_target = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_target = ave_LED_target + ...
                squeeze(LED_target{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_target = ave_LED_target / num_distractors;
        lh = plot(ave_LED_target, '-k');
        set( lh, 'LineWidth', 2 );
        
        ave_LED_distractor_target = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_distractor_target = ave_LED_distractor_target + ...
                squeeze(LED_distractor_target{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_distractor_target = ave_LED_distractor_target / num_distractors;
        lh = plot(ave_LED_distractor_target, '-r');
        set( lh, 'LineWidth', 2 );
        
        axis on
        box off
        ylabel(mask_str_list{1});
        set(gca, 'XTickLabel', '');
        set(gca, 'YTickLabel', '');
        set(gca, 'FontSize', 6);
        legend_str = {'T(T)'; 'T(D)'};
        if exist('uioctave', 'file') && uioctave
            legend(legend_str, 'location', 'NorthOutside');
            legend('left');
        else
            legend(legend_str, 'location', 'Best');
        end
        subplot(num_masks, 2, 2);
        axis( [ 0 max_time -1.2 1.2 ] );
        hold on
        
        ave_LED_target_mask = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_target_mask = ave_LED_target_mask + ...
                squeeze(LED_target_mask{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_target_mask = ave_LED_target_mask / num_distractors;
        lh = plot(ave_LED_target_mask, '-k');
        set( lh, 'LineWidth', 2 );
        
        ave_LED_distractor_mask = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_distractor_mask = ave_LED_distractor_mask + ...
                squeeze(LED_distractor_mask{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_distractor_mask = ave_LED_distractor_mask / num_distractors;
        lh = plot(ave_LED_distractor_mask, '-r');
        set( lh, 'LineWidth', 2 );
        
        ave_LED_target_distractor = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_target_distractor = ave_LED_target_distractor + ...
                squeeze(LED_target_distractor{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_target_distractor = ave_LED_target_distractor / num_distractors;
        lh = plot(ave_LED_target_distractor, '-g');
        set( lh, 'LineWidth', 2 );
        
        ave_LED_distractor = zeros(max_time, 1);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_distractor = ave_LED_distractor + ...
                squeeze(LED_distractor{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_distractor = ave_LED_distractor / num_distractors;
        lh = plot(ave_LED_distractor, '-b');
        set( lh, 'LineWidth', 2 );
        
        axis on
        box off
        set(gca, 'XTickLabel', '');
        set(gca, 'YTickLabel', '');
        set(gca, 'FontSize', 6);
        legend_str = {'M(T)'; 'M(D)'; 'D(T)'; 'D(D)'};
        if exist('uioctave', 'file') && uioctave
            legend(legend_str, 'location', 'NorthOutside');
            legend('left');
        else
            legend(legend_str, 'location', 'Best');
        end
        for i_mask = 2 : num_masks
            subplot(num_masks, 2, 2 * i_mask - 1);
            axis( [ 0 max_time -1.2 1.2 ] );
            hold on
            
            ave_LED_target = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_target = ave_LED_target + ...
                    squeeze(LED_target{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_target = ave_LED_target / num_distractors;
            lh = plot(ave_LED_target, '-k');
            set( lh, 'LineWidth', 2 );
            
            ave_LED_distractor_target = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_distractor_target = ave_LED_distractor_target + ...
                    squeeze(LED_distractor_target{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_distractor_target = ave_LED_distractor_target / num_distractors;
            lh = plot(ave_LED_distractor_target, '-r');
            set( lh, 'LineWidth', 2 );
            
            axis on
            box off
            ylabel(mask_str_list{i_mask});
            set(gca, 'YTickLabel', '');
            if i_mask ~= num_masks
                set(gca, 'XTickLabel', '');
            end
            subplot(num_masks, 2, 2 * i_mask );
            axis( [ 0 max_time -1.2 1.2 ] );
            hold on
            
            ave_LED_target_mask = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_target_mask = ave_LED_target_mask + ...
                    squeeze(LED_target_mask{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_target_mask = ave_LED_target_mask / num_distractors;
            lh = plot(ave_LED_target_mask, '-k');
            set( lh, 'LineWidth', 2 );
            
            ave_LED_distractor_mask = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_distractor_mask = ave_LED_distractor_mask + ...
                    squeeze(LED_distractor_mask{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_distractor_mask = ave_LED_distractor_mask / num_distractors;
            lh = plot(ave_LED_distractor_mask, '-r');
            set( lh, 'LineWidth', 2 );
            
            ave_LED_target_distractor = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_target_distractor = ave_LED_target_distractor + ...
                    squeeze(LED_target_distractor{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_target_distractor = ave_LED_target_distractor / num_distractors;
            lh = plot(ave_LED_target_distractor, '-g');
            set( lh, 'LineWidth', 2 );
            
            ave_LED_distractor = zeros(max_time, 1);
            for i_distractor = start_distractor : stop_distractor
                ave_LED_distractor = ave_LED_distractor + ...
                    squeeze(LED_distractor{i_target,i_distractor,i_mask,i_SoA});
            end
            ave_LED_distractor = ave_LED_distractor / num_distractors;
            lh = plot(ave_LED_distractor, '-b');
            set( lh, 'LineWidth', 2 );
            
            axis on
            if i_mask ~= num_masks
                set(gca, 'XTickLabel', '');
            end
            set(gca, 'YTickLabel', '');
            box off
        end % i_mask
        drawnow();
    end % i_SoA
end % i_target
