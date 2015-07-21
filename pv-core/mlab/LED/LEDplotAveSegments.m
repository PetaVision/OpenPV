function fh_segments = ...
    LEDplotAveSegments( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, mask_str_list, ...
    data_SoA, num_SoAs_data, max_time, ...
    LED_target_segment, LED_distractor_segment)

global num_LED_segs
  global start_target stop_target
  global start_distractor stop_distractor

num_distractors = stop_distractor - start_distractor + 1;
fh_segments = zeros(num_targets, 1, num_SoAs_data);
for i_target = start_target : stop_target
        for i_SoA = 2 : 2 : num_SoAs_data
            i_mask = 1;
            fh_segments(i_target, 1, i_SoA) = ...
                figure( 'Name', ...
                ['Segments vs. time: target (',   target_str_list{i_target}, ...
                '); ', ...
                'SoA = ', num2str(data_SoA(i_SoA)) ] );
            subplot(num_masks, 1, 1);
            axis( [ 0 max_time -1.2 1.2 ] );
            hold on
	    
        ave_LED_target_segment = zeros(max_time, num_LED_segs);
        for i_distractor = start_distractor : stop_distractor
            ave_LED_target_segment = ave_LED_target_segment + ...
                squeeze(LED_target_segment{i_target,i_distractor,i_mask,i_SoA});
        end
        ave_LED_target_segment = ave_LED_target_segment / num_distractors;
        lh = plot(squeeze(ave_LED_target_segment));
            set( lh, 'LineWidth', 2 );
            axis on
            box off
            ylabel(mask_str_list{1});
            set(gca, 'XTickLabel', '');
            set(gca, 'YTickLabel', '');
            set(gca, 'FontSize', 6);
            legend_str = {'1'; '2'; '3'; '4'; '5'; '6'; '7'};
            if exist('uioctave', 'file') && uioctave
                legend(legend_str, 'location', 'NorthOutside');
                legend('left');
            else
                legend(legend_str, 'location', 'Best');
            end
		
            for i_mask = 2 : num_masks
	      
                subplot(num_masks, 1, i_mask );
                axis( [ 0 max_time -1.2 1.2 ] );
                hold on
		ave_LED_target_segment = zeros(max_time, num_LED_segs);
		for i_distractor = start_distractor : stop_distractor
		  ave_LED_target_segment = ave_LED_target_segment + ...
                      squeeze(LED_target_segment{i_target,i_distractor,i_mask,i_SoA});
                end
		ave_LED_target_segment = ave_LED_target_segment / num_distractors;
                lh = plot(squeeze(ave_LED_target_segment));
                set( lh, 'LineWidth', 2 );
                axis on
                box off
                ylabel(mask_str_list{i_mask});
                set(gca, 'YTickLabel', '');
                if i_mask ~= num_masks
                    set(gca, 'XTickLabel', '');
                end
                box off
		    
            end % i_mask
            drawnow();
        end % i_SoA
end % i_target
