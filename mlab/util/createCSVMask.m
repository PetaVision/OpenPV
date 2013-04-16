%%%%%%%%%%%%%%%%%%%%
%% createCSVMask.m
%%   Dylan Paiton
%%   Los Alamos National Laboratory
%%
%% Inputs:
%%   boxes        - created by getCSVBoxes.m; a list of coordinates for bounding-boxes
%%   height       - height of output mask
%%   width        - width of output mask
%%   mask_type    - shape of output clusters; either 'box' or 'ellipse'
%%   (conf_thesh) - optional; only puts boxes above threshold into mask
%%   (sub_box)    - optional; the input is indexed from a traditional 'boxes' variable
%%                            i.e. if boxes = getCSVBoxes(..); then sub_box = boxes(idx);
%% 
%% Outputs:
%%   mask           - logical array of zeros with filled 'clusters' defined by the boxes input
%%   (num_clusters) - optional; number of clusters in mask
%%   (out_struct)   - Struct containing information about the mask and it's inverse
%%   (inv_mask)     - The logical (not matrix) inverse of the mask. (i.e. all 0s are 1s and vice-versa)
%%
%% NOTE: 'boxes' should be output from getCSVBoxes.m
%%
%%%%%%%%%%%%%%%%%%%%

function [varargout] = createCSVMask(boxes, height, width, mask_type, conf_thresh, sub_box)

    %Variable input
    if gt(nargin,6) || lt(nargin,4)
        error('createCSVMask: Incorrect number of input arguments')
    elseif eq(nargin,4)
        conf_thresh = 0;
        sub_box = false;
    elseif eq(nargin,5)
        sub_box = false;
    end

    %Variable output
    nout = max(nargout,1);
    if gt(nout,4)
        error(['createCSVMask: Inappropriate number of output arguments. nout = ',num2str(nout)])
    end

    if sub_box
        max_class_vector = 1;
    else
        max_class_vector = length(boxes);
    end

    mask = zeros([height width]);
    num_clusters = 0;

    if strcmp(mask_type,'box') %Make boxes
        if gt(max_class_vector,0) % If there are clusters
            for i=1:max_class_vector % For each bounding box
                clust_mask = zeros([height width]);

                if sub_box
                    curr_box = boxes;
                else
                    curr_box = boxes(i);
                end

                if isfield(curr_box,'Frame')
                    if strcmp(curr_box.Frame,'N/A')
                        continue
                    end
                end
                if isfield(curr_box,'Confidence')
                    if lt(curr_box.Confidence,conf_thresh)
                        continue
                    end
                end

                num_clusters = num_clusters + 1;

                if isfield(curr_box,'bp')
                    x_pts = [round(curr_box.bp(1,1)) ...
                        round(curr_box.bp(2,1)) ...
                        round(curr_box.bp(3,1)) ...
                        round(curr_box.bp(4,1))];
                    y_pts = [round(curr_box.bp(1,2)) ...
                        round(curr_box.bp(2,2)) ...
                        round(curr_box.bp(3,2)) ...
                        round(curr_box.bp(4,2))];
                elseif isfield(curr_box,'BB_X1')
                    x_pts = [round(curr_box.BB_X1) ...
                        round(curr_box.BB_X2) ...
                        round(curr_box.BB_X3) ...
                        round(curr_box.BB_X4)];
                    y_pts = [round(curr_box.BB_Y1) ...
                        round(curr_box.BB_Y2) ...
                        round(curr_box.BB_Y3) ...
                        round(curr_box.BB_Y4)];
                else
                    error('createCSVMask: ERROR: Unknown struct type.')
                end

                coords1 = [[y_pts(1),x_pts(1)];[y_pts(2),x_pts(2)]];
                coords2 = [[y_pts(2),x_pts(2)];[y_pts(3),x_pts(3)]];
                coords3 = [[y_pts(3),x_pts(3)];[y_pts(4),x_pts(4)]];
                coords4 = [[y_pts(4),x_pts(4)];[y_pts(1),x_pts(1)]];

                clust_mask = bresenham(clust_mask,coords1);
                clust_mask = bresenham(clust_mask,coords2);
                clust_mask = bresenham(clust_mask,coords3);
                clust_mask = bresenham(clust_mask,coords4);

                center = round(curr_box.C); % [x;y] or [width;height]

                [clust_mask idx] = bwfill(clust_mask,center(1),center(2)); % BWFILL wants center as [x y]

                if isfield(curr_box,'cf')
                    clust_mask = clust_mask.*curr_box.cf; %give values a height according to confidence
                end

                mask = mask + clust_mask;
            end
        end
    elseif strcmp(mask_type,'ellipse') %Make Ellipses
        for i=1:max_class_vector
            if sub_box
                curr_box = boxes;
            else
                curr_box = boxes(i);
            end

            [Y,X] = meshgrid(0:width-1,0:height-1);
            X = X - curr_box.C(1);
            Y = Y - curr_box.C(2);
            % A matrix is turned around
            clust_mask = X.^2 * curr_box.A(2,2) + X.*Y * (curr_box.A(1,2) + curr_box.A(2,1)) + Y.^2 * curr_box.A(1,1) < 1;

            num_clusters = num_clusters + 1;

            mask = mask + clust_mask.*curr_box.cf;
        end
    else
        error('createCSVMask: ERROR: mask_type must be either ''ellipse'' or ''box''')
    end

    inv_mask    = logical(~mask);
    [nuy,nux]   = find(mask);
    [inuy,inux] = find(inv_mask);
    num_hits    = length(nuy);
    num_hitsi   = length(inuy);
    nuyx        = [nuy nux];
    inuyx       = [inuy inux];

    out_struct.num_hits  = num_hits;
    out_struct.num_hitsi = num_hitsi;
    out_struct.yx        = nuyx;
    out_struct.iyx       = inuyx;

    switch (nout)
        case 1
            varargout{1} = mask;
        case 2
            varargout{1} = mask;
            varargout{2} = num_clusters;
        case 3
            varargout{1} = mask;
            varargout{2} = num_clusters;
            varargout{3} = out_struct;
        case 4
            varargout{1} = mask;
            varargout{2} = num_clusters;
            varargout{3} = out_struct;
            varargout{4} = inv_mask;
        otherwise
            error(['createCSVMask: Inappropriate number of output arguments. nout = ',num2str(nout)])
    end
end
