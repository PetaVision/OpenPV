function [spike_movie] = stdp_reconstruct_movie( spike_array, layer_list )

if isempty(layer_list)
    layer_list = 1;
end
spike_duration = 1;
first_frame = 180;
last_frame = 812;

num_movie_layers = length(layer_list);
for i_layer = 1 : num_movie_layers
    layer = layer_list(i_layer);
    pv_globals( layer );
    scrsz = get(0,'ScreenSize');
%     fh = figure('Position',[scrsz(3)/4 scrsz(4)/4 scrsz(3)/2 scrsz(4)/2]);
%     fh = figure('Position',[scrsz(3)/4 scrsz(4)/4 560 420]);
    fh = figure('Position',[0 100 560 420]);
    axis tight
    axis off
    box off
    axh = gca;
    set(axh,'nextplot','replacechildren');
    for i_frame = first_frame : last_frame
        plot_title = ['Spike Movie for layer = ', num2str(layer), ': frame = ', num2str(i_frame)];
        recon_array = spike_array{layer}( i_frame:(i_frame+spike_duration-1),: );
        recon_array = sum(recon_array,1);
        recon_array(recon_array > 1) = 1;
        fh = stdp_reconstruct(full(recon_array), plot_title, fh);
        figure(fh);
        spike_movie(i_frame - first_frame + 1) = getframe(fh);
        fh = clf(fh);
    end % i_frame
end % i_layer
close(fh);












