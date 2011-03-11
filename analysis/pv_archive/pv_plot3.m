%%
close all
clear all

% set simulation parameters

begin_step = 21;%201;%

if (ispc)
    output_path = '..\src\output\';
    input_path =  '..\src\io\input\t64_';
    tiff_path =   '..\src\io\input\amoeba2X.tif';
else
    output_path = '../src/output/';
    input_path =  '../src/io/input/t64_';
    tiff_path =   '../src/io/input/amoeba2X.tif';
end

spike_name{1} = 'f1.bin';
spike_name{2} = 'f2.bin';
vmem_name{1} = 'V1.bin';
vmem_name{2} = 'V2.bin';

% Read parameters from file which pv created
load([output_path, 'params.txt'],'-ascii')
NX = params(1);
NY = params(2);
NO = params(3);
NK = params(4);
N = params(5);
DTH  = params(6);
n_time_steps = params(7);
%input_filenae = params(8); % causes problems.
%DK= 1/(sqrt(2)*(NK-1));

%n_time_steps = 100;

%load([input_path, 'inparams.txt'], '-ascii')
%num_fig = circle1_inparams(1);
num_fig=1;

Ni= N/NK;

%read spike events
spike_array = cell(2,1);
spike_array2 = spike_array;
spike_filename = cell(2,1);
spike_filename{1} = [output_path, spike_name{1}];
fid = fopen(spike_filename{1}, 'r', 'native');
[spike_array{1}, num_read] = fread(fid, [N, n_time_steps], 'float');
spike_array{1} = (spike_array{1}(:,begin_step:end))';
fclose(fid);

ave_rate = 1000 * sum(spike_array{1}(:)) / ( length(spike_array{1}(:)) );
disp(['ave_rate{1} = ', num2str(ave_rate)]);

bin_size = 1;

n_steps = size(spike_array{1},1);
pad = mod( n_steps, bin_size );
spike_array_tmp = spike_array{1}';
spike_array_tmp = spike_array_tmp(:,pad+1:n_steps);
n_steps = size(spike_array_tmp,2);
spike_array_tmp = reshape(spike_array_tmp, [N, n_steps / bin_size, bin_size] );
spike_array_tmp = squeeze( sum( spike_array_tmp, 3 ) );
spike_array2{1} = spike_array_tmp';

spike_filename{2} = [output_path, spike_name{2}];
if exist(spike_filename{2},'file')
    fid = fopen(spike_filename{2}, 'r', 'native');
    spike_array{2} = fread(fid, [Ni, n_time_steps], 'float');
    spike_array{2} = (spike_array{2}(:,begin_step:end))';
    fclose(fid);

    ave_ratei = 1000 * sum(spike_array{2}(:)) / ( length(spike_array{2}(:)) );
    disp(['ave_rate{2} = ', num2str(ave_ratei)]);

    n_steps = size(spike_array{2},1);
    pad = mod( n_steps, bin_size );
    spike_array_tmp = spike_array{2}';
    spike_array_tmp = spike_array_tmp(:,pad+1:n_steps);
    n_steps = size(spike_array_tmp,2);
    spike_array_tmp = reshape(spike_array_tmp, [Ni, n_steps / bin_size, bin_size] );
    spike_array_tmp = squeeze( sum( spike_array_tmp, 3 ) );
    spike_array2{2} = spike_array_tmp';
end

%read membrane potentials
vmem_array = cell(2,1);
vmem_filename = cell(2,1);
vmem_filename{1} = [output_path, vmem_name{1}];
if exist(vmem_filename{1},'file')
    fid = fopen(vmem_filename{1}, 'r', 'native');
    [vmem_array{1}, num_read]  = fread(fid, [N, n_time_steps], 'float');
    if num_read ~= N * n_time_steps
        error('num_read ~= N * n_time_steps');
    end
    vmem_array{1} = reshape(vmem_array{1}, [N, n_time_steps]);
    vmem_array{1} = (vmem_array{1}(:, begin_step:end))';
    fclose(fid);
end

vmem_filename{2} = [output_path, vmem_name{2}];
if exist(vmem_filename{2},'file')
    fid = fopen(vmem_filename{2}, 'r', 'native');
    [vmem_array{2},num_read] = fread(fid, [Ni, n_time_steps], 'float'); % fread(fid, inf, 'float32');
    if num_read ~= Ni * n_time_steps
        error('num_read ~= Ni * n_time_steps');
    end
    vmem_array{2} = reshape(vmem_array{2}, [Ni, n_time_steps]);
    vmem_array{2} = (vmem_array{2}(:, begin_step:end))';
    fclose(fid);
end

%read input image
input_filename = 'input.bin';
input_filename = [input_path, input_filename];
if exist(input_filename,'file')
    fid = fopen(input_filename, 'r', 'native');
    input_array = fread(fid, (N), 'float');
    fclose(fid);
end

%read number of objects and number of their indices in image
num_filename = 'num.bin';
num_filename = [input_path, num_filename];
%disp (num_filename);
if exist (num_filename, 'file');
    fid= fopen(num_filename,'r', 'native');
    %num_fig= fread(fid, 1, 'int');
    clutter_ind = fread(fid, 1, 'int');
    num_indices = fread(fid, num_fig,'int');
    fclose(fid);
end

parse_colored_tiff = 1
if (parse_colored_tiff)
    [targ,clutter] = pv_parseTiff( tiff_path );
    % TODO: support for multiple objects--multiple calls to parseTiff?
    figure_ndx{1} = pv_findo( spike_array{1}, targ, n_time_steps, NX, NY, NO, NK);
    figure_ndxi{1} = pv_findo( spike_array{2}, targ, n_time_steps, NX, NY, NO, 1);
    clutter_index = pv_findo( spike_array{1}, clutter, n_time_steps, NX, NY, NO, NK);
    clutter_indexi = pv_findo( spike_array{2}, clutter, n_time_steps, NX, NY, NO, 1);    
else
    %read indices of objects
    figure_ndx= cell(num_fig,1);
    figure_ndxi= figure_ndx;
    for i_fig = 0:num_fig-1
        numstr = int2str(i_fig);
        input_indices_file = 'figure_';
        input_indices_file = [input_indices_file,numstr];
        input_indices_file = [input_indices_file, '.bin'];
        input_indices_file = [input_path,input_indices_file];
        fid = fopen(input_indices_file, 'r', 'native');
        figure_ndx{i_fig+1}= fread(fid, num_indices(i_fig+1),'int');
        figure_ndx{i_fig+1}= figure_ndx{i_fig+1} + 1;
        %figure_ndx{i_fig+1} = figure_ndx{i_fig+1}(:NK:end);
        figure_ndxi{i_fig+1}= fix((figure_ndx{i_fig+1} - 1)/NK)+1;
        fclose(fid);
    end

    %read indices of clutter
    clutter_filename = 'clutter.bin';
    clutter_filename = [input_path, clutter_filename];
    cfid= fopen(clutter_filename, 'r', 'native');
    clutter_index = fread (cfid, clutter_ind, 'int');
    clutter_index = clutter_index + 1;
    fclose(cfid);
end

%print out rate
for i_fig=1:num_fig

    figure_rate = ...
        1000 * sum(sum(spike_array{1}(:,figure_ndx{i_fig}))) / ...
        (length(figure_ndx{i_fig}) * size(spike_array{1},1) );
    disp(['figure', num2str(i_fig),'_rate{1} = ', num2str(figure_rate)]);
    figure_ratei = ...
        1000* sum(sum(spike_array{2}(:,figure_ndxi{i_fig}))) / ...
        (length(figure_ndxi{i_fig})* size(spike_array{i_fig},1));
    disp(['figure',num2str(i_fig),'_rate{2} = ', num2str(figure_ratei)]);
end
clutter_rate = ...
    1000 * sum(sum(spike_array{1}(:,clutter_index))) / ...
    (length(clutter_index) * size(spike_array{1},1) );
disp(['clutter_rate{1} = ', num2str(clutter_rate)]);


%%
%n_time_steps = size(spike_array{1},1);
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
    axis([0 n_steps 0 N]);
    hold on
    % box on
    axis tight
    [spike_id, spike_time, spike_val] = find((spike_array{1})');
    %plot(spike_time, spike_id, '.k');

    for i_fig=1:num_fig
        [spike_id, spike_time, spike_val] = find((spike_array{1}(:,figure_ndx{i_fig}))');
        plot(spike_time, figure_ndx{i_fig}(spike_id), '.r');
    end
end

%plot PSTH for just clutter first
% plot_title = ['Histogram for clutter'];
% figure('Name',plot_title);
cPSTH = 1000 * sum(spike_array2{1}(:,clutter_index),2)/(length(clutter_index)*bin_size);
% lh = plot(cPSTH, '-k');
% set(lh, 'LineWidth', 2);
% axis tight

%plot autocorr for clutter
% plot_title = ['Autocorrelation function for clutter '];
% figure('Name',plot_title);
% maxlag = fix(n_steps/(2*bin_size));
% autocorrc = xcorr(cPSTH, maxlag, 'unbiased');
% autocorrc = ( autocorrc - mean(cPSTH)^2 ) / (mean(cPSTH)+(mean(cPSTH)==0))^2;
% plot((-maxlag:maxlag)*bin_size, autocorrc, '-k');
% axis([-maxlag*bin_size,maxlag*bin_size, min(autocorrc(:)), max(autocorrc(:))+(max(autocorrc(:))==0)]);

%clutter power spectrum
% plot_title = ['Power spectrum for clutter '];
% figure('Name',plot_title);
% freq = 1000*(0:n_steps-1)/n_steps;
% min_ndx = find(freq > 400, 1,'first');
% fft_cPSTH = fft(cPSTH);
% plot(freq(2:min_ndx),...
%     abs(fft_cPSTH(2:min_ndx))/max(1,abs(fft_cPSTH(1))), '-k');
% axis tight

%plot mass PSTH (firing rate histogram combining all cells)
for i_fig=1:num_fig
    if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
        plot_title = ['Histogram for object ',int2str(i_fig)];
        figure('Name',plot_title);
        lh = plot((1:bin_size:n_steps),cPSTH, '-b');
        axis tight
        hold on
        if ~isempty(spike_array{1})
            mPSTH = 1000 * sum(spike_array2{1},2)/(N*bin_size);
            lh = plot((1:bin_size:n_steps),mPSTH, '--k');
            set(lh, 'LineWidth', 2);
            figurePSTH = ...
                1000 * sum(spike_array2{1}(:,figure_ndx{i_fig}),2)/(length(figure_ndx{i_fig})*bin_size);
            plot((1:bin_size:n_steps),figurePSTH, '-k');
            hold on
        end
        if ~isempty(spike_array{2})
            mPSTHi = 1000 * sum(spike_array2{2},2)/(Ni*bin_size);
            lh = plot((1:bin_size:n_steps),mPSTHi, '--r');
            set(lh, 'LineWidth', 2);
            figurePSTHi = ...
                1000 * sum(spike_array2{2}(:,figure_ndxi{i_fig}),2)/(length(figure_ndxi{i_fig})*bin_size);
            plot((1:bin_size:n_steps),figurePSTHi, '-r');
            axis tight
        end
    end



    %plot auto correlation function for each figure

    plot_title = ['Correlation functions for object ',int2str(i_fig)];
    figure('Name',plot_title);
    maxlag= fix(n_steps/(2*bin_size));

    autocorr = xcorr( figurePSTH, maxlag, 'unbiased' );
    autocorri = xcorr( figurePSTHi, maxlag, 'unbiased' );
    autocorr = ( autocorr - mean(figurePSTH)^2 ) / (mean(figurePSTH)+(mean(figurePSTH)==0))^2;
    autocorri = ( autocorri - mean(figurePSTHi)^2 ) / (mean(figurePSTHi)+(mean(figurePSTHi)==0))^2;
    plot((-maxlag:maxlag)*bin_size, autocorr, '-k');
    hold on
    plot((-maxlag:maxlag)*bin_size, autocorri, '-r');
    axis([-maxlag*bin_size,maxlag*bin_size, min(autocorr(:)), max(autocorr(:))+(max(autocorr(:))==0)]);
    hold on
    crosscorr = xcorr(figurePSTH, cPSTH, maxlag, 'unbiased');
    crosscorr = ( crosscorr - mean(cPSTH)*mean(figurePSTH) ) / ...
        ( mean(cPSTH)*mean(figurePSTH) + (mean(cPSTH)*mean(figurePSTH) == 0 ));  
    plot((-maxlag:maxlag)*bin_size, crosscorr, '--k');
    hold on
    if i_fig ~= num_fig
        for j_fig=i_fig+1:num_fig
            plot_title = ['Cross correlation functions for objects ',int2str(i_fig)];
            plot_title = [plot_title,' and '];
            plot_title = [plot_title, int2str(j_fig)];
            figure('Name',plot_title);
            figure2PSTH = 1000 * sum(spike_array{1}(:,figure_ndx{j_fig}),2)/length(figure_ndx{j_fig});
            figure2PSTHi = 1000 * sum(spike_array{2}(:,figure_ndxi{i_fig}),2)/length(figure_ndxi{i_fig});
            crosscorr=xcorr(figurePSTH,figure2PSTH);
            plot(crosscorr,'-r');
            %hold on
            %cosscorri=xcorr(figurePSTHi,figure2PSTHi);
            % plot(crosscorri,'-r');
        end
    end
    axis tight
    % cross corr fo figure with clutter
    %     plot_title = ['Crosscorrelation function for object ',int2str(i_fig)];
    %     figure('Name',plot_title);
    %     crosscorr = xcorr(figurePSTH, cPSTH, maxlag, 'unbiased');
    % crosscorri = xcorr(figurePSTHi, mPSTHi, maxlag, 'unbiased');
%     crosscorr = (crosscorr - mean(cPSTH)*mean(figurePSTH))/(mean(cPSTH)*mean(figurePSTH));
    %crosscorri = (crosscorri - mean(mPSTHi)^2)/mean(mPSTHi);
%     plot((-maxlag:maxlag)*bin_size, crosscorr, '-k');
%     %hold on
    %plot(-maxlag:maxlag, crosscorri, '-r');
%     axis([-maxlag*bin_size, maxlag*bin_size, min(crosscorr(:)), max(crosscorr(:))]);
%     axis tight

    %plot cross corr of circle with everything
    %     plot_title = ['Crosscorrelation function for object ',int2str(i_fig)];
    %     figure('Name',plot_title);
    %     crosscorr = xcorr(figurePSTH, mPSTH, maxlag, 'unbiased');
    %     crosscorri = xcorr(figurePSTHi, mPSTHi, maxlag, 'unbiased');
    %     crosscorr = (crosscorr - mean(mPSTH)^2)/mean(mPSTH);
    %     crosscorri = (crosscorri - mean(mPSTHi)^2)/mean(mPSTHi);
    %     plot(-maxlag:maxlag, crosscorr, '-k');
    %     hold on
    %     plot(-maxlag:maxlag, crosscorri, '-r');
    %     axis([-maxlag, maxlag, min(crosscorri(:)), max(crosscorri(:))]);




    % plot power spectrum of mPSTH
    if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
        plot_title = ['Power spectrum for object ',int2str(i_fig)];
        figure('Name',plot_title);
        freq = 1000*(0:n_steps-1)/n_steps;
        min_ndx = find(freq > 400, 1,'first');
        %min_ndx = min(20,fix(length(freq)/2));
        if ~isempty(spike_array{1})
            %                 fft_mPSTH = fft(mPSTH);
            %                 plot(freq(2:min_ndx),...
            %                     abs(fft_mPSTH(2:min_ndx))/max(1,abs(fft_mPSTH(1))), '--k');
            fft_cPSTH = fft(cPSTH);
            fft_figurePSTH = fft(figurePSTH);
            plot(freq(2:min_ndx),...
                abs(fft_cPSTH(2:min_ndx))/max(1,abs(fft_cPSTH(1))), '--k');
            hold on
            plot(freq(2:min_ndx),...
                abs(fft_figurePSTH(2:min_ndx))/max(1,abs(fft_figurePSTH(1))), '-k');
            hold on
            axis tight
        end
        %         if ~isempty(spike_array{2})
        %             fft_mPSTHi = fft(mPSTHi);
        %             plot(freq(2:min_ndx),...
        %                 abs(fft_mPSTHi(2:min_ndx))/max(1,abs(fft_mPSTHi(1))), '--r');
        %             fft_figurePSTHi = fft(figurePSTHi);
        %             plot(freq(2:min_ndx),...
        %                 abs(fft_figurePSTHi(2:min_ndx))/max(1,abs(fft_figurePSTHi(1))), '-r');
        %             hold on
        %         end
    end
end





% plot reconstructed image
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    plot_title = ['Reconstruction using rate'];
    figure ('Name',plot_title);
    if ~isempty(spike_array2{1})
        rate_array = 1000 * sum(spike_array2{1},1) / ( n_steps );
        max_rate = max(rate_array(:));
        edge_len = sqrt(2)/2;
        max_line_width = 3;
        axis([-1 NX -1 NY]);
        axis square
        % box ON
        hold on;
        rate3D = reshape(rate_array, [NK, NO, NX, NY]);
        for i_k = 1:NK
            for i_theta = 0:NO-1
                delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
                delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
                for i_x = 1:NX
                    for i_y = 1:NY
                        if rate3D(i_k, i_theta+1,i_x,i_y) < ave_rate
                            continue;
                        end
                        lh = line( [i_x - delta_x, i_x + delta_x]', ...
                            [i_y - delta_y, i_y + delta_y]' );
                        line_width = 0.05 + ...
                            max_line_width * rate3D( i_k, i_theta+1,i_x,i_y) / ...
                            max(1000/(N*n_steps), max_rate);
                        set( lh, 'LineWidth', line_width );
                        line_color = 1 - rate3D( i_k, i_theta+1,i_x,i_y) / ...
                            max(1000/(N*n_time_steps), max_rate);
                        set( lh, 'Color', line_color*[1 1 1]);
                    end
                end
            end
        end
    end
    plotinhibit = 1;
    if plotinhibit==1

        if ~isempty(spike_array{2})
            plot_title = ['Reconstruction of inhibition using rate'];
            figure ('Name',plot_title);
            edge_len = sqrt(2)/2;
            max_line_width = 3;
            axis([-1 NX -1 NY]);
            axis square
            % box ON
            hold on;
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
                            max(1000/(Ni*n_time_steps), max_ratei);
                        %if (line_width < 0.0)
                        %   line_width = 0.0;
                        set( lh, 'LineWidth', line_width );
                        line_color = 1 - rate3Di(i_theta+1,i_x,i_y) /  ...
                            max(1000/(Ni*n_time_steps), max_ratei);
                        set( lh, 'Color', line_color*[1 0 0]);
                    end
                end
            end
        end
    end
end

%% reconstrunction using power
if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
    if ~isempty(spike_array{1})
        plot_title = ['Reconstruction using power'];
       figure ('Name',plot_title);
         num_steps_excite = size(spike_array{1},1);
        freq_excite = 1000*(0:num_steps_excite-1)/num_steps_excite;
        min_ndx = find(freq_excite >= 60, 1,'first');
        max_ndx = find(freq_excite <= 90, 1,'last');
        fft_array = abs(fft(spike_array{1})).^2;
        peak_power= max(fft_array(min_ndx:max_ndx,:));
        %peak_power = peak_power ./ (fft_array(1,:) + (fft_array(1,:)==0));
        edge_len = sqrt(2)/2;
        max_line_width = 3;
        axis([-1 NX -1 NY]);
        axis square
        % box ON
        hold on;
        ave_power = mean(peak_power(:));
        max_power = max(peak_power(:));
        peak_power = reshape(peak_power, [NK, NO, NX, NY]);
        for i_k = 1:NK
            for i_theta = 0:NO-1
                delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
                delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
                for i_x = 1:NX
                    for i_y = 1:NY
                        if  peak_power(i_k, i_theta+1,i_x,i_y) < ave_power
                            continue;
                        end
                        lh = line( [i_x - delta_x, i_x + delta_x]', ...
                            [i_y - delta_y, i_y + delta_y]' );
                        line_width = 0.05 + ...
                            max_line_width * peak_power( i_k, i_theta+1,i_x,i_y) / ...
                            ((max_power==0) + max_power);
                        set( lh, 'LineWidth', line_width );
                        line_color = 1 - peak_power( i_k, i_theta+1,i_x,i_y) / ...
                            ((max_power==0) + max_power);
                        set( lh, 'Color', line_color*[1 1 1]);
                    end
                end
            end
        end
    end
    if ~isempty(spike_array{2})
        plot_title = ['Reconstruction of inhibition using power'];
       figure ('Name',plot_title);
        num_steps_inhibit = size(spike_array{2},1);
        freq_inhibit = 1000*(0:num_steps_inhibit-1)/num_steps_inhibit;
        min_ndx = find(freq_inhibit >= 60, 1,'first');
        max_ndx = find(freq_inhibit <= 90, 1,'last');
        fft_array_inhibit = abs(fft(spike_array{2})).^2;
        peak_power_inhibit= max(fft_array_inhibit(min_ndx:max_ndx,:));
        %peak_power = peak_power_inhibit ./ (fft_array_inhibit(1,:) + (fft_array_inhibit(1,:)==0));
        edge_len = sqrt(2)/2;
        max_line_width = 3;
        axis([-1 NX -1 NY]);
        axis square
        % box ON
        hold on;
        ave_power_inhibit = mean(peak_power_inhibit(:));
        max_power_inhibit = max(peak_power_inhibit(:));
        peak_power_inhibit = reshape(peak_power_inhibit, [NO, NX, NY]);
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if  peak_power_inhibit(i_theta+1,i_x,i_y) < ave_power_inhibit
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [i_y - delta_y, i_y + delta_y]' );
                    line_width = 0.05 + ...
                        max_line_width * peak_power_inhibit( i_theta+1,i_x,i_y) / ...
                        ((max_power_inhibit==0) + max_power_inhibit);
                    set( lh, 'LineWidth', line_width );
                    line_color = 1 - peak_power_inhibit( i_theta+1,i_x,i_y) / ...
                        ((max_power_inhibit==0) + max_power_inhibit);
                    set( lh, 'Color', line_color*[1 1 1]);
                end
            end
        end
    end
end

%% eigen reconstruction

ave_power = mean(peak_power(find(peak_power > 0)));
rate_mask = find(rate3D > ave_rate);
power_mask = find(peak_power > ave_power);

disp([num2str(length(power_mask)), ' in power_mask greater than ', num2str(ave_power)])
%power_mask = rate_mask;
num_mask = length(power_mask);
num_steps_corr = size(spike_array{1},1);
maxlag= fix(num_steps_corr/4);
%pack
freq_xcorr = 1000*(0:(2*maxlag))/(2*maxlag+1);
min_ndx = find(freq_xcorr >= 60, 1,'first');
max_ndx = find(freq_xcorr <= 90, 1,'last');

sparse_row_ndx = repmat(power_mask,1,num_mask)';
sparse_row_ndx = sparse_row_ndx(:);
sparse_col_ndx = repmat(power_mask,num_mask,1);
sparse_col_ndx = sparse_col_ndx(:);

sparse_corr = sparse(sparse_row_ndx, sparse_col_ndx, 0, N, N, num_mask*num_mask);

stride = 1;
for pre=1:stride:num_mask
    for post=1:stride:num_mask
%
%        temp = xcorr( spike_array{1}(:,power_mask(pre:pre+stride)), spike_array{1}(:,power_mask(post:post+stride)), maxlag, 'unbiased' );
        temp = xcorr( spike_array{1}(:,power_mask(pre)), spike_array{1}(:,power_mask(post)), maxlag, 'unbiased' );
        
        temp = abs(fft(temp));
        temp = max(temp(min_ndx:max_ndx,:));
       
        sparse_corr(power_mask(pre),power_mask(post)) = temp;
    end
    if (mod(pre, 50) == 0)
        disp(pre);
    end
end

%sparse_corr = xcorr( spike_array{1}(:,power_mask), maxlag, 'unbiased' );
%sparse_corr = abs(fft(spare_corr));
%sparse_corr = max(sparse_corr(min_ndx:max_ndx,:));
%sparse_corr = sparse(sparse_row_ndx, sparse_col_ndx, sparse_corr, N, N, num_mask^2);

[row, col] = find(sparse_corr);
meancorr = mean(sparse_corr(sparse_corr~=0));
for idx = 1:size(row)
    sparse_corr(row(idx), col(idx)) = sparse_corr(row(idx), col(idx)) - meancorr;
    if (mod(idx, 50) == 0)
        disp(pre);
    end
end

sparse_corr = 0.5 * (sparse_corr + sparse_corr');

options.issym = 1;
num_eigen = 1;
options.v0 = peak_power(:);
[eigen_vec,eigen_value, eigen_flag] = eigs(sparse_corr, num_eigen, 'lm', options);
for i_vec = 1:num_eigen
    plot_title = ['Reconstruction using eigenvector ', num2str(i_vec)];
    figure ('Name',plot_title);
    edge_len = sqrt(2)/2;
    max_line_width = 3;
    axis([-1 NX -1 NY]);
    axis square
    box ON
    hold on;
    largest_eigen_vec = reshape(eigen_vec(:,i_vec), [NK, NO, NX, NY]);
    if mean(eigen_vec(power_mask,i_vec)) < 0
        largest_eigen_vec = -largest_eigen_vec;
    end
    max_eigen = max(largest_eigen_vec(:));
    for i_k = 1:NK
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if  largest_eigen_vec(i_k, i_theta+1,i_x,i_y) <= 0
                        continue;
                    end
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [i_y - delta_y, i_y + delta_y]' );
                    line_width = 0.05 + ...
                        max_line_width * largest_eigen_vec( i_k, i_theta+1,i_x,i_y) / ...
                        ((max_eigen==0) + max_eigen);
                    set( lh, 'LineWidth', line_width );
                    line_color = 1 - largest_eigen_vec( i_k, i_theta+1,i_x,i_y) / ...
                        ((max_eigen==0) + max_eigen);
                    set( lh, 'Color', line_color*[1 1 1]);
                end
            end
        end
    end
end

%% plot maximum stimulated membrane potential
if ~isempty(spike_array2{1}) || ~isempty(spike_array2{2})
    figure
    offset_id = 0;%2*NK;
    if ~isempty(spike_array2{1})
        [max_vmem, max_id] = max(rate_array);
        max_id = max_id+offset_id;
        plot(vmem_array{1}(:,max_id), '-k');
        axis tight;
        hold on;
        [spike_times, spike_id, spike_vals] = find(spike_array2{1}(:,max_id));
        lh = line( [max(1,spike_times-1), max(1,spike_times-1)]', ...
            [vmem_array{1}(max(1,spike_times-1), max_id), 1.0*spike_vals]' );
        set(lh, 'Color', [0 0 0]);
    end
    if ~isempty(spike_array{2})
        plot(vmem_array{2}(:,fix((max_id-1)/NK)+1), '-r');
        hold on;
        [spike_timesi, spike_idi, spike_valsi] = find(spike_array2{2}(:,fix((max_id-1)/NK)+1));
        lh = line( [max(1,spike_timesi-1), max(1,spike_timesi-1)]', ...
            [vmem_array{2}(max(1,spike_timesi-1), fix((max_id-1)/NK)+1), 1.0*spike_valsi]' );
        set(lh, 'Color', [1 0 0]);
    end
end


plotinput=1;
if plotinput==1
    %plot raw input image
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
    input3D = reshape(input_array, [NK, NO, NX, NY]);
    for i_k = 1: NK
        for i_theta = 0:NO-1
            delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
            delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
            for i_x = 1:NX
                for i_y = 1:NY
                    if input3D(i_k,i_theta+1,i_x,i_y) <= ave_input
                        continue;
                    end
                    %plot(i_x, i_y, '.k');
                    lh = line( [i_x - delta_x, i_x + delta_x]', ...
                        [i_y - delta_y, i_y + delta_y]' );
                    line_width = 0.05 + ...
                        max_line_width * ...
                        ( input3D(i_k,i_theta+1,i_x,i_y) - min_input ) / ...
                        ( max_input - min_input );
                    set( lh, 'LineWidth', line_width );
                    line_color = 1 - ...
                        ( input3D(i_k,i_theta+1,i_x,i_y) - min_input ) / ...
                        ( max_input - min_input );
                    set( lh, 'Color', line_color*[1 1 1]);
                end
            end
        end
    end
end
%
% %%
%
% plot "weights" (typically after turning on just one neuron)
plot_weights = 0;
if plot_weights == 1
    figure;
    weight3D = reshape(vmem_array(1,:), [NK, NO, NX, NY]);
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
%
%
%
% %%
%
% save_mat_file = 0
% if save_mat_file == 1
%     save_file = 'pv_Ex025_In025_NAmp05_NFrq_05_circwcluter01';
%     save_file = [output_path, save_file];
%     save( save_file, 'spike_array', 'vmem_array', 'input_array');
% end
% %%
%
% % make movie out of spikes
% play_movie = 0;
% if play_movie
%     figure;
%     numFrames=n_time_steps;
%     A=moviein(numFrames);
%     set(gca,'NextPlot','replacechildren');
%     if ~isempty(spike_array{1})
%         edge_len = sqrt(2)/2;
%         max_line_width = 3;
%          for which_time=1:numFrames
%
%             % need to do the following inside loop due to clf
%             axis([-1 NX -1 NY]);
%             axis square
%             box ON
%             hold on;
%
%             spike3D = reshape(spike_array{1}(which_time,:), [NO, NX, NY]);
%             for i_theta = 0:NO-1
%                 delta_x = edge_len * ( cos(i_theta * DTH * pi / 180 ) );
%                 delta_y = edge_len * ( sin(i_theta * DTH * pi / 180 ) );
%                 for i_x = 1:NX
%                     for i_y = 1:NY
%                         if spike3D(i_theta+1,i_x,i_y) == 0
%                             continue;
%                         end
%                         lh = line( [i_x - delta_x, i_x + delta_x]', ...
%                             [i_y - delta_y, i_y + delta_y]' );
%                         line_width = max_line_width;
%                         set( lh, 'LineWidth', line_width );
%                         line_color = 0;
%                         set( lh, 'Color', line_color*[1 1 1]);
%
%                         % Progress bar for movie at bottom of plot
%                         % Ignore scale for progress bar.
%                         lh = line( [which_time/numFrames*NX, ...
%                             which_time/numFrames*NX]', [-1 0]');
%                         set (lh, 'LineWidth', 1);
%                         set (lh, 'Color', [0 0 1.0]);
%                     end  % y
%                 end % x
%             end % orientation
%
%             A(:,which_time)=getframe;
%             clf;
%         end % frame
%
%         axis([-1 NX -1 NY]);
%         axis square
%         box ON
%         movie(A,10) % play movie ten times
%     end
% end