%%%%%%%%%%%%%%%%%%%%
%% getCSVBoxes.m
%%   Dylan Paiton
%%   Los Alamos National Laboratory
%%
%% Inputs:
%%   csv_file - exact path to CSV file to make boxes from
%%   (object) - Object that it is looking for. If no object is specified, it will do all objects.
%%   (frame)  - If you wish to get boxes for a specific frame
%% 
%% Outputs:
%%   boxes         - [1 x num_csv_lines] struct array with bounding box location and information
%%   num_truth_BBs - If CSV file has other objects besides specified object, this will return the number of boxes with specified object
%%   num_CSV_lines - Total number of lines in input CSV file
%%   num_frames    - Total number of frames represented in CSV file
%%                   NOTE: frames in CSV file may not be contiguous
%%
%%   NOTE: CSV file should be in the format:
%%         Frame,BoundingBox_X1,BoundingBox_Y1,BoundingBox_X2,BoundingBox_Y2,BoundingBox_X3,BoundingBox_Y3,BoundingBox_X4,BoundingBox_Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version
%%
%%%%%%%%%%%%%%%%%%%%
function [boxes num_truth_BBs num_CSV_lines num_frames] = getCSVBoxes(csv_file,object,frame)

    if gt(nargin,3) || lt(nargin,1)
        error('getCSVBoxes: Incorrect number of input arguments')
    elseif eq(nargin,1)
        object = 'PV_ALL';
        check_frame = 0;
    elseif eq(nargin,2)
        check_frame = 0;
    else %3 arguments
        check_frame = 1;
    end

    if ~exist(csv_file,'file')
        error(['~exist: csv_file = ',csv_file])
    end

    fid      = fopen(csv_file,'r');
    header   = fgets(fid);
    csv_list = cell(1);

    i_CSV = 0;
    while ~feof(fid)
        i_CSV = i_CSV + 1;
        csv_list{i_CSV} = fgets(fid);
    end
    fclose(fid);

    num_CSV_lines = size(csv_list,2);

    boxes(num_CSV_lines).Frame      = [1, 2];
    boxes(num_CSV_lines).BB_X1      = [2, 3];
    boxes(num_CSV_lines).BB_Y1      = [3, 4];
    boxes(num_CSV_lines).BB_X2      = [4, 5];
    boxes(num_CSV_lines).BB_Y2      = [5, 6];
    boxes(num_CSV_lines).BB_X3      = [6, 7];
    boxes(num_CSV_lines).BB_Y3      = [7, 8];
    boxes(num_CSV_lines).BB_X4      = [8, 9];
    boxes(num_CSV_lines).BB_Y4      = [9, 10];
    boxes(num_CSV_lines).C          = [10, 11];
    boxes(num_CSV_lines).ObjectType = [11, 12];
    boxes(num_CSV_lines).Confidence = [12, 13];
    boxes(num_CSV_lines).Info       = [13, 14];

    num_truth_BBs = 0;
    num_frames = 0;
    for i_CSV = 1:num_CSV_lines
        csv_vals = regexp(csv_list{i_CSV},',','split'); 
        compute_boxes = 0;
        if check_frame
            if eq(str2num(csv_vals{1,1}),frame)
                if strcmp(csv_vals{1,10},object) || strcmp(object,'PV_ALL')
                    compute_boxes = 1;
                end
            end
        else
            if strcmp(csv_vals{1,10},object) || strcmp(object,'PV_ALL')
                compute_boxes = 1;
            end
        end

        if compute_boxes
            num_truth_BBs = num_truth_BBs + 1;
            boxes(i_CSV).Frame      = str2num(csv_vals{1,1});
            boxes(i_CSV).BB_X1      = str2num(csv_vals{1,2});
            boxes(i_CSV).BB_Y1      = str2num(csv_vals{1,3});
            boxes(i_CSV).BB_X2      = str2num(csv_vals{1,4});
            boxes(i_CSV).BB_Y2      = str2num(csv_vals{1,5});
            boxes(i_CSV).BB_X3      = str2num(csv_vals{1,6});
            boxes(i_CSV).BB_Y3      = str2num(csv_vals{1,7});
            boxes(i_CSV).BB_X4      = str2num(csv_vals{1,8});
            boxes(i_CSV).BB_Y4      = str2num(csv_vals{1,9});
            boxes(i_CSV).ObjectType = csv_vals{1,10};
            boxes(i_CSV).Confidence = str2num(csv_vals{1,13});
            boxes(i_CSV).Info       = str2num(csv_vals{1,14});

            %Don't allow edge to be 0
            if eq(boxes(i_CSV).BB_X1,0)
                boxes(i_CSV).BB_X1 = 1;
            end
            if eq(boxes(i_CSV).BB_X2,0)
                boxes(i_CSV).BB_X2 = 1;
            end
            if eq(boxes(i_CSV).BB_X3,0)
                boxes(i_CSV).BB_X3 = 1;
            end
            if eq(boxes(i_CSV).BB_X4,0)
                boxes(i_CSV).BB_X4 = 1;
            end
            if eq(boxes(i_CSV).BB_Y1,0)
                boxes(i_CSV).BB_Y1 = 1;
            end
            if eq(boxes(i_CSV).BB_Y2,0)
                boxes(i_CSV).BB_Y2 = 1;
            end
            if eq(boxes(i_CSV).BB_Y3,0)
                boxes(i_CSV).BB_Y3 = 1;
            end
            if eq(boxes(i_CSV).BB_Y4,0)
                boxes(i_CSV).BB_Y4 = 1;
            end

            x_pts = [boxes(i_CSV).BB_X1 ...
                boxes(i_CSV).BB_X2 ...
                boxes(i_CSV).BB_X3 ...
                boxes(i_CSV).BB_X4];
            y_pts = [boxes(i_CSV).BB_Y1 ...
                boxes(i_CSV).BB_Y2 ...
                boxes(i_CSV).BB_Y3 ...
                boxes(i_CSV).BB_Y4];

            [min_x_val min_x_idx] = min(x_pts);
            [max_x_val max_x_idx] = max(x_pts);
            [min_y_val min_y_idx] = min(y_pts);
            [max_y_val max_y_idx] = max(y_pts);

            boxes(i_CSV).C        = [min_x_val+(max_x_val-min_x_val)/2;min_y_val+(max_y_val-min_y_val)/2];

            if boxes(i_CSV).Frame > num_frames
                num_frames = boxes(i_CSV).Frame;
            end
       else
           boxes(i_CSV).Frame      = 'N/A';
           boxes(i_CSV).BB_X1      = 'N/A';
           boxes(i_CSV).BB_Y1      = 'N/A';
           boxes(i_CSV).BB_X2      = 'N/A';
           boxes(i_CSV).BB_Y2      = 'N/A';
           boxes(i_CSV).BB_X3      = 'N/A';
           boxes(i_CSV).BB_Y3      = 'N/A';
           boxes(i_CSV).BB_X4      = 'N/A';
           boxes(i_CSV).BB_Y4      = 'N/A';
           boxes(i_CSV).C          = 'N/A';
           boxes(i_CSV).ObjectType = 'N/A';
           boxes(i_CSV).Confidence = 'N/A';
           boxes(i_CSV).Info       = 'N/A';
       end
    end
    num_frames = num_frames + 1; % 0 indexed, gotta add one to get the number of frames
end
