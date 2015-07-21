function fh_segments = ...
    LEDplotSegments( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, mask_str_list, ...
    data_SoA, num_SoAs_data, max_time, ...
    LED_target_segment, LED_distractor_segment)

  global start_target stop_target
  global start_distractor stop_distractor

fh_segments = zeros(num_targets, num_distractors, num_SoAs_data);
for i_target = start_target : stop_target
    for i_distractor = start_distractor : stop_distractor
        for i_SoA = 1 : 1 : num_SoAs_data
            i_mask = 1;
            fh_segments(i_target, i_distractor, i_SoA) = ...
                figure( 'Name', ...
                ['Segments vs. time: target (',   target_str_list{i_target}, ...
                ') distractor (',   distractor_str_list{i_distractor}, ...
                '); ', ...
                'SoA = ', num2str(data_SoA(i_SoA)) ] );
            subplot(num_masks, 2, 1);
            axis( [ 0 max_time -2 2 ] );
            hold on
            lh = plot(squeeze(LED_target_segment{i_target, i_distractor, i_mask, i_SoA}));
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
            subplot(num_masks, 2, 2);
            axis( [ 0 max_time -2 2 ] );
            hold on
            lh = plot(squeeze(LED_distractor_segment{i_target, i_distractor, i_mask, i_SoA}));
            set( lh, 'LineWidth', 2 );
            axis on
            box off
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
                subplot(num_masks, 2, 2 * i_mask - 1);
                axis( [ 0 max_time -2 2 ] );
                hold on
                lh = plot(squeeze(LED_target_segment{i_target, i_distractor, i_mask, i_SoA}));
                set( lh, 'LineWidth', 2 );
                axis on
                box off
                ylabel(mask_str_list{i_mask});
                set(gca, 'YTickLabel', '');
                if i_mask ~= num_masks
                    set(gca, 'XTickLabel', '');
                end
                subplot(num_masks, 2, 2 * i_mask );
                axis( [ 0 max_time -2 2 ] );
                hold on
                lh = plot(squeeze(LED_distractor_segment{i_target, i_distractor, i_mask, i_SoA}));
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
    end % i_distractor
end % i_target
