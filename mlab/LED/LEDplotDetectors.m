function fh_detector = ...
    LEDplotDetectors( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, mask_str_list, ...
    data_SoA, num_SoAs_data, max_time, ...
    LED_target, LED_distractor_target, ...
    LED_target_mask, LED_distractor_mask, ...
		     LED_distractor, LED_target_distractor)

  global start_target stop_target
  global start_distractor stop_distractor

fh_detector = zeros(num_targets, num_distractors, num_SoAs_data);
for i_target = start_target : stop_target
    for i_distractor = start_target : stop_distractor
        for i_SoA = 1 : 1 : num_SoAs_data
            i_mask = 1;
            fh_detector(i_target, i_distractor, i_SoA) = ...
                figure( 'Name', ...
                ['target (',   target_str_list{i_target}, ...
                ') distractor (',   distractor_str_list{i_distractor}, ...
                ') vs. time; ', ...
                'SoA = ', num2str(data_SoA(i_SoA)) ] );
            subplot(num_masks, 2, 1);
            axis( [ 0 max_time -2 2 ] );
            hold on
            lh = plot(squeeze(LED_target{i_target,i_distractor,i_mask,i_SoA}), '-k');
            set( lh, 'LineWidth', 2 );
            lh = plot(squeeze(LED_distractor_target{i_target,i_distractor,i_mask,i_SoA}), '-r');
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
            axis( [ 0 max_time -2 2 ] );
            hold on
            lh = plot(squeeze(LED_target_mask{i_target,i_distractor,i_mask,i_SoA}), '-k');
            set( lh, 'LineWidth', 2 );
            lh = plot(squeeze(LED_distractor_mask{i_target,i_distractor,i_mask,i_SoA}), '-r');
            set( lh, 'LineWidth', 2 );
            lh = plot(squeeze(LED_target_distractor{i_target,i_distractor,i_mask,i_SoA}), '-g');
            set( lh, 'LineWidth', 2 );
            lh = plot(squeeze(LED_distractor{i_target,i_distractor,i_mask,i_SoA}), '-b');
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
                axis( [ 0 max_time -2 2 ] );
                hold on
                lh = plot(squeeze(LED_target{i_target,i_distractor,i_mask,i_SoA}), '-k');
                set( lh, 'LineWidth', 2 );
                lh = plot(squeeze(LED_distractor_target{i_target,i_distractor,i_mask,i_SoA}), '-r');
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
                lh = plot(squeeze(LED_target_mask{i_target,i_distractor,i_mask,i_SoA}), '-k');
                set( lh, 'LineWidth', 2 );
                lh = plot(squeeze(LED_distractor_mask{i_target,i_distractor,i_mask,i_SoA}), '-r');
                set( lh, 'LineWidth', 2 );
                lh = plot(squeeze(LED_target_distractor{i_target,i_distractor,i_mask,i_SoA}), '-g');
                set( lh, 'LineWidth', 2 );
                lh = plot(squeeze(LED_distractor{i_target,i_distractor,i_mask,i_SoA}), '-b');
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
