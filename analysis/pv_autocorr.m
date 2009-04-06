function [] = pv_psth()

    global N NO NX NY DTH n_time_steps begin_step output_path input_path
    global spike_array i_fig num_fig rate_array

    %plot mass PSTH (firing rate histogram combining all cells)
    for i_fig=1:num_fig
        if ~isempty(spike_array{1}) || ~isempty(spike_array{2})
            plot_title = ['Histogram for object ',int2str(i_fig)];
            figure('Name',plot_title);
            if ~isempty(spike_array{1})
                mPSTH = 1000 * sum(spike_array{1},2)/N;
                lh = plot(mPSTH, '--k');
                set(lh, 'LineWidth', 2);
                hold on
                figurePSTH = 1000 * sum(spike_array{1}(:,figure_ndx{i_fig}),2)/length(figure_ndx{i_fig});
                plot(figurePSTH, '-k');
                hold on
            end
            if ~isempty(spike_array{2})
                mPSTHi = 1000 * sum(spike_array{2},2)/N;
                lh = plot(mPSTHi, '--r');
                set(lh, 'LineWidth', 2);
                figurePSTHi = 1000 * sum(spike_array{2}(:,figure_ndx{i_fig}),2)/length(figure_ndx{i_fig});
                plot(figurePSTHi, '-r');
            end
        end
    end

    %plot auto correlation function for each figure
    plot_title = ['Autocorrelation function for object ',int2str(i_fig)];
    figure('Name',plot_title);
    title(['Autocorrelation function for object ',int2str(i_fig)]);
    maxlag= floor(n_time_steps/2); 	% needs to be an integer for Octave
    
    autocorr=xcorr(figurePSTH, maxlag, 'unbiased');
    autocorri=xcorr(figurePSTHi, maxlag, 'unbiased');
    autocorr= (autocorr- mean(figurePSTH)^2)/mean(figurePSTH);
    autocorri= (autocorri- mean(figurePSTHi)^2)/mean(figurePSTHi);
    plot(-maxlag:maxlag, autocorr, '-k');
    hold on
    plot(-maxlag:maxlag, autocorri, '-r');
    axis([-maxlag,maxlag, min(autocorr(:)), max(autocorr(:))]);
    
    %plot cross corr of circle with everything
    plot_title = ['Crosscorrelation function for object ',int2str(i_fig)];
    figure('Name',plot_title);
    crosscorr = xcorr(figurePSTH, mPSTH, maxlag, 'unbiased'); 
    crosscorri = xcorr(figurePSTHi, mPSTHi, maxlag, 'unbiased');
    crosscorr = (crosscorr - mean(mPSTH)^2)/mean(mPSTH);
    crosscorri = (crosscorri - mean(mPSTHi)^2)/mean(mPSTHi);
    plot(-maxlag:maxlag, crosscorr, '-k');
    hold on
    plot(-maxlag:maxlag, crosscorri, '-r');
    axis([-maxlag, maxlag, min(crosscorri(:)), max(crosscorri(:))]);

    % plot power spectrum of mPSTH
    if ~isempty(spike_array{1}) || ~isempty(spike_array{2})

        plot_title = ['Power spectrum for object ',int2str(i_fig)];
        figure('Name',plot_title);
        freq = 1000*(0:n_time_steps-1)/n_time_steps;
        min_ndx = find(freq > 160, 1,'first');
        %min_ndx = min(20,fix(length(freq)/2));
        if ~isempty(spike_array{1})
            fft_mPSTH = fft(mPSTH);
            plot(freq(2:min_ndx),...
                abs(fft_mPSTH(2:min_ndx))/max(1,abs(fft_mPSTH(1))), '--k');
            fft_figurePSTH = fft(figurePSTH);
            plot(freq(2:min_ndx),...
                abs(fft_figurePSTH(2:min_ndx))/max(1,abs(fft_figurePSTH(1))), '-k');
            hold on
        end
        if ~isempty(spike_array{2})
            fft_mPSTHi = fft(mPSTHi);
            plot(freq(2:min_ndx),...
                abs(fft_mPSTHi(2:min_ndx))/max(1,abs(fft_mPSTHi(1))), '--r');
            fft_figurePSTHi = fft(figurePSTHi);
            plot(freq(2:min_ndx),...
                abs(fft_figurePSTHi(2:min_ndx))/max(1,abs(fft_figurePSTHi(1))), '-r');
            hold on
        end
    end
    if i_fig ~= num_fig
        for j_fig=i_fig+1:num_fig
            figure;
            plot_title = ['Cross correlation functions for objects ',int2str(i_fig)];
            plot_title = [plot_title,' and '];
            plot_title = [plot_title, int2str(j_fig)];
            figure('Name',plot_title);
            figure2PSTH = 1000 * sum(spike_array{1}(:,figure_ndx{j_fig}),2)/length(figure_ndx{j_fig});
            figure2PSTHi = 1000 * sum(spike_array{2}(:,figure_ndx{i_fig}),2)/length(figure_ndx{i_fig});
            crosscorr=xcorr(figurePSTH,figure2PSTH);
            plot(crosscorr,'-k');
            hold on
            cosscorri=xcorr(figurePSTHi,figure2PSTHi);
            plot(crosscorri,'-r');
        end
    end
end

end
