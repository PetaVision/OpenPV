function fh_performance = ...
    LEDplotModelPerformance( num_targets, ...
    num_distractors, ...
    num_masks, mask_str_list, ...
    num_subjects, subjects_list, LED_prob_correct, data_SoA, max_SoA, fh_performance)

  global start_target stop_target
  global start_distractor stop_distractor

  for i_target = start_target : stop_target
    for i_distractor = start_distractor : stop_distractor
        figure( fh_performance(i_target, i_distractor) );
        j_subject = 0;
        for i_subject = subjects_list
            j_subject = j_subject + 1;
            hold on
            lh_performance = zeros(num_masks, 1);
            ah_subjects = subplot( num_subjects, 2, 2 * j_subject );
            axis( [ 0 max_SoA 0 1 ] );
            set( ah_subjects, 'YTickLabel', [] );
            if j_subject == num_subjects
                xlabel('SoA (msec)');
            else
                set( ah_subjects, 'XTickLabel', [] );
            end
            hold on
            line_color = get( gca, 'ColorOrder' );
            num_color = length(line_color);
            for i_mask = 1 : num_masks
                lh_performance(i_mask) = ...
                    plot( data_SoA, squeeze( LED_prob_correct{ i_target, i_distractor, i_mask } ) );
                set( lh_performance(i_mask), 'Color', line_color( mod( i_mask - 1, num_color ) + 1, : ) );
                set( lh_performance(i_mask), 'LineWidth', 2 );
                set( lh_performance(i_mask), 'LineStyle', '-');
            end % i_mask
        end % i_subject
%         subplot( num_subjects, 2, 2);
        if exist('uioctave', 'file') && uioctave
            legend(mask_str_list, 'location', 'NorthOutside');
            legend('left');
        else
            legend(lh_performance, mask_str_list, 'location', 'Best');
        end
        drawnow();
    end % i_distractor
end % i_target
