function [w f] = filterDoG(wOn, wOff, n_cells, img_size)
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

    %Add borders
    for v=img_size:img_size:(sqrt(n_cells)-1)*img_size %Loop over cells
        RFs(v,:) = -1;
        RFs(:,v) = -1;
        RF2s(v,:) = 0;
        RF2s(:,v) = 0;
    end

    f=figure
    imagesc(RFs+RF2s);
    colormap gray
    axis off          % Remove axis ticks and numbers
    axis image        % Set aspect ratio to obtain square pixels

    w = RFs+RF2s;

end



