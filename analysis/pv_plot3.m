%%
close all
clear all

% set simulation parameters
begin_step = 3;%201;%

%output_path = 'C:\cygwin\home\gkenyon\syn_cogn\output\';
%output_path = 'C:\cygwin\home\gkenyon\PetaVision\src\output\';
%output_path = 'C:\cygwin\home\gkenyon\trunk\src\output\';
output_path = 'C:\Users\admin\linux\petavision\trunk\src\output\'; % Dan

% Read parameters from file which pv created
load([output_path, 'params.txt'],'-ascii')
NX = params(1);
NY = params(2);
NO = params(3);
N = params(4);
DTH  = params(5)
n_time_steps = params(6);

%read spike events
spike_array = cell(2,1);
spike_filename = cell(2,1);
spike_filename{1} = 'f0.bin';
spike_filename{1} = [output_path, spike_filename{1}];
fid = fopen(spike_filename{1}, 'r', 'native');
spike_array{1} = fread(fid, [N, n_time_steps+begin_step-1], 'float');
spike_array{1} = (spike_array{1}(:,begin_step:end))';
fclose(fid);

ave_rate = 1000 * sum(spike_array{1}(:)) / ( N * n_time_steps );
disp(['ave_rate{1} = ', num2str(ave_rate)]);

spike_filename{2} = 'h0.bin';
spike_filename{2} = [output_path, spike_filename{2}];
if exist(spike_filename{2},'file')
    fid = fopen(spike_filename, 'r', 'native');
    spike_array{2} = fread(fid, [N, n_time_steps+begin_step-1], 'float');
    spike_array{2} = (spike_array{2}(:,begin_step:end))';
    fclose(fid);

    ave_ratei = 1000 * sum(spike_array{2}(:)) / ( N * n_time_steps );
    disp(['ave_rate{2} = ', num2str(ave_ratei)]);
end

%read membrane potentials
vmem_array = cell(2,1);
vmem_filename = cell(2,1);
vmem_filename{1} = 'V0.bin';
vmem_filename{1} = [output_path, vmem_filename{1}];
if exist(vmem_filename{1},'file')
    fid = fopen(vmem_filename{1}, 'r', 'native');
    vmem_array{1} = fread(fid, [N, n_time_steps+begin_step-1], 'float');
    vmem_array{1} = (vmem_array{1}(:,begin_step:end))';
    fclose(fid);
end

vmem_filename{2} = 'Vinh0.bin';
vmem_filename{2} = [output_path, vmem_filename{2}];
if exist(vmem_filename{2},'file')
    fid = fopen(vmem_filename{2}, 'r', 'native');
    vmem_array{2} = fread(fid, [N, n_time_steps+begin_step-1], 'float');
    vmem_array{2} = (vmem_array{2}(:,begin_step:end))';
    fclose(fid);
end

%read input image
input_filename = 'input_0.bin';
input_filename = [output_path, input_filename];
if exist(input_filename,'file')
    fid = fopen(input_filename, 'r', 'native');
    input_array = fread(fid, N, 'float');
    fclose(fid);
end

circle_ndx = [2727,2728,2736,2744,2753,2999,3057,3270,3362,3550,...
    3658,4117,4243,4685,4827,4972,5116,5260,5404,5548,...
    5692,5835,5981,6419,6549,7002,7118,7298,7398,7601,...
    7671,7905,7912,7920,7928] + 1;


circle_rate = 1000 * sum(sum(spike_array{1}(:,circle_ndx))) / ...
    ( length(circle_ndx) * size(spike_array{1},1) );
disp(['circle_rate{1} = ', num2str(circle_rate)]);


%%
n_time_steps = size(spike_array{1},1);
if ~exist('NO', 'var')
    NO = 8;
end

% These params should already be set?
%N = size(spike_array{1},2);
%NX = floor( sqrt( N/NO ) );
%NY = NX;
%DTH  = 180./NO;

% raster plot
if ~isempty(spike_array{1})
    figure;
    axis([0 n_time_steps 0 N]);
    hold on
    box on
    [spike_id, spike_time, spike_val] = find((spike_array{1})');
    plot(spike_time, spike_id, '.k');
    [spike_id, spike_time, spike_val] = find((spike_array{1}(:,circle_ndx))');
    plot(spike_time, circle_ndx(spike_id), '.r');
end

%plot mass PSTH (firing rate histogram combining all cells)
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    figure;
    if ~isempty(spike_array{1})
        mPSTH = 1000 * sum(spike_array{1},2)/N;
        lh = plot(mPSTH, '--k');
        set(lh, 'LineWidth', 2);
        hold on
        circlePSTH = 1000 * sum(spike_array{1}(:,circle_ndx),2)/length(circle_ndx);
        plot(circlePSTH, '-k');
        hold on
    end
    if ~isempty(spike_array{2})
        mPSTHi = 1000 * sum(spike_array{2},2)/N;
        lh = plot(mPSTHi, '--r');
        set(lh, 'LineWidth', 2);
        circlePSTHi = 1000 * sum(spike_array{2}(:,circle_ndx),2)/length(circle_ndx);
        plot(circlePSTHi, '-r');
    end
end

% plot power spectrum of mPSTH
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    figure;
    freq = 1000*(0:n_time_steps-1)/n_time_steps;
    min_ndx = min(20,fix(length(freq)/2));
    if ~isempty(spike_array{1})
        fft_mPSTH = fft(mPSTH);
        plot(freq(2:min_ndx),...
            abs(fft_mPSTH(2:min_ndx))/max(1,abs(fft_mPSTH(1))), '--k');
        fft_circlePSTH = fft(circlePSTH);
        plot(freq(2:min_ndx),...
            abs(fft_circlePSTH(2:min_ndx))/max(1,abs(fft_circlePSTH(1))), '-k');
        hold on
    end
    if ~isempty(spike_array{2})
        fft_mPSTHi = fft(mPSTHi);
        plot(freq(2:min_ndx),...
            abs(fft_mPSTHi(2:min_ndx))/max(1,abs(fft_mPSTHi(1))), '--r');
        fft_circlePSTHi = fft(circlePSTHi);
        plot(freq(2:min_ndx),...
            abs(fft_circlePSTHi(2:min_ndx))/max(1,abs(fft_circlePSTHi(1))), '-r');
        hold on
    end
end

% plot reconstructed image
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    figure;
    if ~isempty(spike_array{1})
        rate_array = 1000 * sum(spike_array{1},1) / ( n_time_steps );
        max_rate = max(rate_array(:));
        edge_len = sqrt(2)/2;
        max_line_width = 3;
        axis([-1 NX -1 NY]);
        axis square
        box ON
        hold on;
        rate3D = reshape(rate_array, [NO, NX, NY]);
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if rate3D(i_theta+1,i_x,i_y) < ave_rate
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [i_y - delta_y, i_y + delta_y]' );
                    line_width = 0.05 + ...
                        max_line_width * rate3D(i_theta+1,i_x,i_y) / ...
                        max(1000/(N*n_time_steps), max_rate);
                    set( lh, 'LineWidth', line_width );
                    line_color = 1 - rate3D(i_theta+1,i_x,i_y) / ...
                        max(1000/(N*n_time_steps), max_rate);
                    set( lh, 'Color', line_color*[1 1 1]);
                end
            end
        end
    end

    if ~isempty(spike_array{2})
        rate_arrayi = 1000 * sum(spike_array{2},1) / ( n_time_steps );
        max_ratei = max(rate_arrayi(:));
        rate3Di = reshape(rate_arrayi, [NO, NX, NY]);
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if rate3Di(i_theta+1,i_x,i_y) <= ave_ratei
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [i_y - delta_y, i_y + delta_y]' );
                    line_width = 0.05 + ...
                        max_line_width * rate3Di(i_theta+1,i_x,i_y) /  ...
                        max(1000/(N*n_time_steps), max_ratei);
                    set( lh, 'LineWidth', line_width );
                    line_color = 1 - rate3Di(i_theta+1,i_x,i_y) /  ...
                        max(1000/(N*n_time_steps), max_ratei);
                    set( lh, 'Color', line_color*[1 0 0]);
                end
            end
        end
    end
end

% plot maximum stimulated membrane potential
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    figure
    if ~isempty(spike_array{1})
        [max_vmem, max_id] = max(rate_array);
        plot(vmem_array{1}(:,max_id), '-k');
        hold on;
        [spike_times, spike_id, spike_vals] = find(spike_array{1}(:,max_id));
        lh = line( [max(1,spike_times-1), max(1,spike_times-1)]', ...
            [vmem_array{1}(max(1,spike_times-1), max_id), 1.0*spike_vals]' );
        set(lh, 'Color', [0 0 0]);
    end
    if ~isempty(spike_array{2})
        plot(vmem_array{2}(:,max_id), '-r');
        hold on;
        [spike_times, spike_id, spike_vals] = find(spike_array{2}(:,max_id));
        lh = line( [max(1,spike_times-1), max(1,spike_times-1)]', ...
            [vmem_array{2}(max(1,spike_times-1), max_id), 1.0*spike_vals]' );
        set(lh, 'Color', [1 0 0]);
    end
end

% plot raw input image
figure;
axis([-1 NX -1 NY]);
axis square
box ON
hold on;
ave_input = sum( input_array(:) ) / ( N );
disp(['ave_input = ', num2str(ave_input)]);
max_input = max(input_array(:));
min_input = min(input_array(:));
edge_len = sqrt(2)/2;
max_line_width = 3;
input3D = reshape(input_array, [NO, NX, NY]);
for i_theta = 0:NO-1
    delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
    delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
    for i_x = 1:NX
        for i_y = 1:NY
            if input3D(i_theta+1,i_x,i_y) <= ave_input
                continue;
            end
            %plot(i_x, i_y, '.k');
            lh = line( [i_x - delta_x, i_x + delta_x]', ...
                [i_y - delta_y, i_y + delta_y]' );
            line_width = 0.05 + ...
                max_line_width * ...
                ( input3D(i_theta+1,i_x,i_y) - min_input ) / ...
                ( max_input - min_input );
            set( lh, 'LineWidth', line_width );
            line_color = 1 - ...
                ( input3D(i_theta+1,i_x,i_y) - min_input ) / ...
                ( max_input - min_input );
            set( lh, 'Color', line_color*[1 1 1]);
        end
    end
end




%%

% plot "weights" (typically after turning on just one neuron)
plot_weights = 0;
if plot_weights == 1
    figure;
    weight3D = reshape(vmem_array(2,:), [NO, NX, NY]);
    i_x0 = fix(NX/2) + 1;
    i_y0 = fix(NY/2) + 1;
    i_theta0 = 2;
    k0 = i_theta0+1 + i_x0 * NO + i_y0 * NX * NO;
    weight3D(i_theta0 + 1, i_x0, i_y0) = 0.0;
    min_weight = min(weight3D(:));
    %weight3D = weight3D - min_weight;
    ave_weight = mean(weight3D(:));
    max_weight = max(weight3D(:));
    edge_len = sqrt(2)/2;
    max_line_width = 3;
    axis([-1 NX -1 NY]);
    axis square;
    box ON
    hold on;
    delta_x = edge_len * ( cos(i_theta0 * DTH * pi / 180 ) );
    delta_y = edge_len * ( sin(i_theta0 * DTH * pi / 180 ) );
    lh = line( [i_x0 - delta_x, i_x0 + delta_x]', ...
        [i_y0 - delta_y, i_y0 + delta_y]' );
    line_width = max_line_width;
    set( lh, 'LineWidth', line_width );
    set( lh, 'Color', [1 0 0]);
    for i_theta = 0:NO-1
        delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
        delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
        for i_x = 1:NX
            for i_y = 1:NY
                if weight3D(i_theta+1,i_x,i_y) < 0
                    continue;
                end
                lh = line( [i_x - delta_x, i_x + delta_x]', ...
                    [i_y - delta_y, i_y + delta_y]' );
                line_width = 0.05 + ...
                    max_line_width * (weight3D(i_theta+1,i_x,i_y) - 0) / (max_weight - 0);
                set( lh, 'LineWidth', line_width );
                line_color = (weight3D(i_theta+1,i_x,i_y) - 0) / (max_weight - 0);
                set( lh, 'Color', (1-line_color) * [1 1 1]);
            end
        end
    end
end



%%

save_mat_file = 0
if save_mat_file == 1
    save_file = 'pv_Ex025_In025_NAmp05_NFrq_05_circwcluter01';
    save_file = [output_path, save_file];
    save( save_file, 'spike_array', 'vmem_array', 'input_array');
end
%%

% make movie out of spikes
play_movie = 0;
if play_movie
    figure;
    numFrames=n_time_steps;
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

            spike3D = reshape(spike_array{1}(which_time,:), [NO, NX, NY]);
            for i_theta = 0:NO-1
                delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
                delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
                for i_x = 1:NX
                    for i_y = 1:NY
                        if spike3D(i_theta+1,i_x,i_y) == 0
                            continue;
                        end
                        lh = line( [i_x - delta_x, i_x + delta_x]', ...
                            [i_y - delta_y, i_y + delta_y]' );
                        line_width = max_line_width;
                        set( lh, 'LineWidth', line_width );
                        line_color = 0;
                        set( lh, 'Color', line_color*[1 1 1]);
                        
                        % Progress bar for movie at bottom of plot
                        % Ignore scale for progress bar.
                        lh = line( [which_time/numFrames*NX, ...
                            which_time/numFrames*NX]', [-1 0]');
                        set (lh, 'LineWidth', 1);
                        set (lh, 'Color', [0 0 1.0]);
                    end  % y
                end % x
            end % orientation
            
            A(:,which_time)=getframe;
            clf;
        end % frame
        
        axis([-1 NX -1 NY]);
        axis square
        box ON
        movie(A,10) % play movie ten times
    end
end
