function  stdp_plotSpikes( spike_array, index_array)
% plotSpikes plots spikes as vertical lines
%   Detailed explanation goes here
global input_dir  NX NY 

lh = figure('Name', 'Raster plot of spike times');
hold on
color_order = get(gca,'ColorOrder');

num_max = 7;

if isempty(index_array)
    
    index_array = 1:10: num_max;
    
else
    
    num_max = min(7,length(index_array));
    
end

spike_array2 = spike_array( :, index_array );

bin_size = 1000; % average over 1 second window
tot_steps = size( spike_array2, 1 );
num_bins = fix( tot_steps / bin_size );
edges = 1:bin_size:tot_steps;


for n=1:num_max
    [spike_times, spike_id, spike_vals] = ...
        find(spike_array2(:,n));  % find returns [row,col,v]
    rate(n) = length(spike_times);
    lh = line( [spike_times, spike_times]', ...
        [n-1, n]', 'Color',color_order(n,:) );
    set(lh, 'LineWidth', 2.0);
    if n == 1
       xlabel('Time (msec)'); 
    end
    display( [ 'neuron:', num2str(n), ' rate = ', ...
        num2str( 1000*rate(n) / tot_steps ) ] );
    pause(0.1)
    psth{n} = histc(spike_times, edges);
    
end

%compute and plot averaged rate
lh = figure('Name', 'Moving window average rate');
hold on
for n=1:num_max
    lf = subplot(num_max, 1 , n);
    lf = plot(psth{n},'-o','MarkerEdgeColor',color_order(n,:),...
                'MarkerFaceColor',color_order(n,:));
   
    xlabel('Time (sec)'); 
   
    
end