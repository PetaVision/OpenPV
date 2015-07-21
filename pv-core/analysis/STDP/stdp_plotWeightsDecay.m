function [fh] = stdp_plotWeightsDecay(w_file, ind, NX, NY)

%fprintf('Compute weight decay for NX = %d NY = %d\n',NX,NY);

ind = ind - 1; % to get kx and ky we neeed indexing starting from 0
sym={'o-g','o-r'}; % synapsez connected to active retina pixels or not
                   % not yet implemented


for i=1:length(ind)
    k = ind(i); % linear neuron index 
    kx = mod(k,NX); 
    ky = (k - kx)/NX;
    fprintf(' k = %d kx = %d ky = %d \n',k+1,kx+1,ky+1);
    %pause
    
    % shift kx and ky to Matlab indexing
    plot_title = ['Weights evolution for neuron (' num2str(kx+1) ',' num2str(ky+1) ')'];
    [PATCH, patch_size, NXP, NYP] = stdp_plotPatch(w_file, kx+1, ky+1, NX, NY, plot_title );
    fprintf('NXP = %d NYP = %d\n',NXP,NYP);    
    %size(PATCH)
    figure('Name', plot_title);
    for k=1:patch_size,
        %plot(PATCH(:,k),sym{k}),hold on,
        %plot(PATCH(:,k),sym{plot_symbol(k)+1}),hold on,
        plot(PATCH(:,k),'-b'),hold on,
    end

    AVERAGE_PATCH = reshape(mean(PATCH,1),[NXP NYP]);
    figure('Name', ['Average weights for neuron (' num2str(kx+1) ',' num2str(ky+1) ')']);
    imagesc(AVERAGE_PATCH', 'CDataMapping','direct');
    % NOTE: It seems that I need to transpose PATCH_RATE
    colorbar
    
    % normalize weights vector patch and compute overlap 
    % with first weights vector patch
    V0 = PATCH(1,:) ./ norm(PATCH(1,:));
    plot_title = ['Weights decay for neuron (' num2str(kx+1) ',' num2str(ky+1) ')'];
    figure('Name', plot_title);
    for k=2:size(PATCH,1),
        V = PATCH(k,:) ./ norm(PATCH(k,:));
        O(k-1) = V0 * V';
    end
    
    plot(O,'o-b');
    pause
end



