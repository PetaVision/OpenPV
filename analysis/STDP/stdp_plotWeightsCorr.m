function [fh] = stdp_plotWeightsCorr(w_file, ind, NX, NY, S, plot_title)
% S is the length of each autocorrelation sequence

fprintf('%s\n', plot_title);
fprintf('Compute weight correlations for NX = %d NY = %d\n',NX,NY);
pause

ind = ind - 1; % to get kx and ky we neeed indexing starting from 0
sym={'o-g','o-r'}; % synapsez connected to active retina pixels or not
                   % not yet implemented
                   
for i=1:length(ind)
    k = ind(i); % linear neuron index 
    kx = mod(k,NX); 
    ky = (k - kx)/NX;
    fprintf(' k = %d kx = %d ky = %d \n',k+1,kx+1,ky+1);
    pause
    
    % shift kx and ky to Matlab indexing
    plot_title = ['Weights evolution for neuron (' num2str(kx+1) ',' num2str(ky+1) ')'];
    [PATCH, patch_size, NXP, NYP] = stdp_plotPatch(w_file, kx+1, ky+1, NX, NY, plot_title );
    size(PATCH)
    fh = figure('Name', plot_title);
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
    
    % compute covariance matrix
    
    C = cov(PATCH);
    AC = diag(R);
    AC = reshape(AC,[NXP NYP]);
    figure('Name', ['Weight correlation matrix (' num2str(kx+1) ',' num2str(ky+1) ')']);
    imagesc(AC', 'CDataMapping','direct');
    % NOTE: It seems that I need to transpose PATCH_RATE
    colorbar
    
    % compute and plot autocorrelation sequences
    figure('Name', ['Weight correlation sequences (' num2str(kx+1) ',' num2str(ky+1) ')']);
    for k=1:patch_size
        v = compCorr(PATCH(:,k),S); % returns a vector of size m
        plot([1:S]', v,'-r'), hold on
    end
    
    pause
end



