
function fh_performance = ...
    LEDplotAveSubjectPerformance( num_targets, target_str_list, ...
    num_distractors, distractor_str_list, ...
    num_masks, ...
    num_subjects, subjects_list, allsubjects_data, data_SoA, max_SoA)

global plot_marker
global start_target stop_target
global start_distractor stop_distractor


fh_performance = zeros(num_targets, 1);
for i_target = start_target : stop_target
    fh_performance(i_target, 1) = ...
        figure( 'Name', ...
        ['Performance vs. SoA', ...
        ': target = ', target_str_list{i_target}] );
    ah_subjects = subplot( 1, 2, 1 );
    axis( [ 0 max_SoA 0 1.05 ] );
    hold on
    xlabel('SoA (msec)');
    line_color = get( gca, 'ColorOrder' );
    num_color = length(line_color);
    lh_subjects = zeros(num_masks, 1);
    for i_mask = 1 : num_masks
        ave_subject = ...
            allsubjects_data( :, i_mask, i_target, :, : );
        ave_subject = squeeze( ave_subject );
        ave_subject = mean( ave_subject, 3 );
        ave_subject = squeeze( ave_subject );
        ave_subject = mean( ave_subject, 2 );
        lh_subjects(i_mask) = ...
            plot( data_SoA, ave_subject );
        set( lh_subjects(i_mask), 'Color', line_color( mod( i_mask - 1, num_color ) + 1, : ) );
        set( lh_subjects(i_mask), 'LineWidth', 2 );
        set( lh_subjects(i_mask), 'MarkerSize', 3 );
        set( lh_subjects(i_mask), 'LineStyle', '-');
        set( lh_subjects(i_mask), 'Marker',plot_marker{i_mask} );
    end % i_mask
end % i_target
