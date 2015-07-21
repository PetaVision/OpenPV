
function fh_performance = ...
    LEDplotSubjectsPerformance( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, ...
    num_subjects, subjects_list, allsubjects_data, data_SoA, max_SoA)

global plot_marker
  global start_target stop_target
  global start_distractor stop_distractor

fh_performance = zeros(num_targets, num_distractors);
for i_target = start_target : stop_target
    for i_distractor = start_distractor : stop_distractor
        fh_performance(i_target, i_distractor) = ...
            figure( 'Name', ...
            ['Performance vs. SoA', ...
            ': target = ', target_str_list{i_target}, ...
            ', distractor = ', distractor_str_list{i_distractor}] );
        j_subject = 0;
        for i_subject = subjects_list
            j_subject = j_subject + 1;
            hold on
            lh_subjects = zeros(num_masks, 1);
            ah_subjects = subplot( num_subjects, 2, 2 * j_subject - 1 );
            axis( [ 0 max_SoA 0 1 ] );
            hold on
            if i_subject == num_subjects
                xlabel('SoA (msec)');
            else
                set( ah_subjects, 'XTickLabel', [] );
            end
            line_color = get( gca, 'ColorOrder' );
            num_color = length(line_color);
            for i_mask = 1 : num_masks
                lh_subjects(i_mask) = ...
                    plot( data_SoA, squeeze( allsubjects_data( :, i_mask, i_target, i_distractor, i_subject ) ) );
                set( lh_subjects(i_mask), 'Color', line_color( mod( i_mask - 1, num_color ) + 1, : ) );
                set( lh_subjects(i_mask), 'LineWidth', 2 );
                set( lh_subjects(i_mask), 'MarkerSize', 3 );
                set( lh_subjects(i_mask), 'LineStyle', '-');
                set( lh_subjects(i_mask), 'Marker',plot_marker{i_mask} );
            end
        end % i_subject
    end % i_distractor
end % i_target
