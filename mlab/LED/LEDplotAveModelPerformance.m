function fh_performance = ...
    LEDplotAveModelPerformance( num_targets, ...
    num_distractors, ...
    num_masks, mask_str_list, ...
    num_subjects, subjects_list, LED_prob_correct, data_SoA, max_SoA, fh_performance)

global start_target stop_target
global start_distractor stop_distractor

num_distractors = stop_distractor - start_distractor + 1;
num_SoA = length(data_SoA);
for i_target = start_target : stop_target
    figure( fh_performance(i_target, 1) );
    ah_subjects = subplot( 1, 2, 2 );
    lh_performance = zeros(num_masks, 1);
    axis( [ 0 max_SoA 0 1.05 ] );
    hold on
    %set( ah_subjects, 'YTickLabel', [] );
    xlabel('SoA (msec)');
    line_color = get( gca, 'ColorOrder' );
    num_color = length(line_color);
    for i_mask = 1 : num_masks
        ave_correct = zeros(num_SoA,1);
        for i_distractor = start_distractor : stop_distractor
            ave_correct = ave_correct + ...
                squeeze( LED_prob_correct{ i_target, i_distractor, i_mask } );
        end % i_distractor
        ave_correct = ave_correct / num_distractors;
        lh_performance(i_mask) = ...
            plot( data_SoA,  ave_correct);
        set( lh_performance(i_mask), 'Color', line_color( mod( i_mask - 1, num_color ) + 1, : ) );
        set( lh_performance(i_mask), 'LineWidth', 2 );
        set( lh_performance(i_mask), 'LineStyle', '-');
    end % i_mask
    if exist('uioctave', 'file') && uioctave
        legend(mask_str_list, 'location', 'NorthOutside');
        legend('left');
    else
        legend(lh_performance, mask_str_list, 'location', 'Best');
    end
    drawnow();
end % i_target
