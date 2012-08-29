function wM = filterDoG(wOn, wOff, n_cells)
    figure
    %clear result

    sigmaC = .7;
    sigmaS = .8/.15;
    sz = round(2*sigmaS);
    dog = DoG(sz,sigmaC,sigmaS);

    %N = sqrt(size(weight,1)/2); % RF size
    RFs = zeros(size(wOn,1), size(wOn,2));
    RF2s = zeros(size(wOn,1), size(wOn,2));

%    for n=1:n_cells
        RFs = RFs + conv2(wOn, 1*dog, 'same');
        if(~isempty(wOff))
            RF2s = RF2s + conv2(wOff, -1*dog, 'same');
        end
%    end
    imagesc(RFs);
    colormap gray
    axis off          % Remove axis ticks and numbers
    axis image        % Set aspect ratio to obtain square pixels

figure
imagesc(RF2s);
colormap gray
axis off          % Remove axis ticks and numbers
axis image        % Set aspect ratio to obtain square pixels

figure
imagesc(RFs+RF2s);
colormap gray
axis off          % Remove axis ticks and numbers
axis image        % Set aspect ratio to obtain square pixels

end
