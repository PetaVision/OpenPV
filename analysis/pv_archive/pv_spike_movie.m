function [] = pv_spike_movie()
 % make movie out of spikes
 play_movie = 0;
 if play_movie
     figure;
     numFrames=n_time_steps-begin_step+1;
     A=moviein(numFrames);
     set(gca,'NextPlot','replacechildren');
     if ~isempty(spike_array{1})
         edge_len = sqrt(2)/2;
         max_line_width = 3;
          for which_time=1:numFrames

             % need to do the following inside loop due to clf
             axis([-1 NX -1 NY]);
             axis square
             box ON
             hold on;

             spike3D = reshape(spike_array{1}(which_time,:), [NK, NO, NX, NY]);
             for i_nk = 1:NK
                 for i_theta = 0:NO-1
                     delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
                     delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
                     for i_x = 1:NX
                         for i_y = 1:NY
                             if spike3D(i_nk, i_theta+1,i_x,i_y) == 0
                                 continue;
                             end
                             lh = line( [i_x - delta_x, i_x + delta_x]', ...
                                 [i_y - delta_y, i_y + delta_y]' );
                             line_width = max_line_width;
                             set( lh, 'LineWidth', line_width );
                             line_color = 0;
                             set( lh, 'Color', i_nk/NK*[0 1 0]);

                             % Progress bar for movie at bottom of plot
                             % Ignore scale for progress bar.
                             lh = line( [which_time/numFrames*NX, ...
                                 which_time/numFrames*NX]', [-1 0]');
                             set (lh, 'LineWidth', 1);
                             set (lh, 'Color', [0 0 1.0]);
                         end  % y
                     end % x
                 end % orientation
             end % nk
             A(:,which_time)=getframe;
             clf;
         end % frame

         axis([-1 NX -1 NY]);
         axis square
         box ON
         movie(A,10) % play movie ten times
     end
 end
end
