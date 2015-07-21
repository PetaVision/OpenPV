function A = stdp_plotRateField(fname, xScale, yScale, Xtarg, Ytarg)
% plot the evolution of the layer firing rate 
% Xtarg and Ytarg contain the X and Y coordinates of the target
% xScale and yScale are scale factors for this layer

% We plot a square around the neurons that have the same receptive
% field. xShare and yShare define the size of the layer patch that
% contains neurons that have the same receptive.

global input_dir  NX NY 

filename = fname;
filename = [input_dir, filename];
    
xShare = 4; % define the size of the layer patch that 
yShare = 4; % contains neurons that have the same receptive field.

NXlayer = NX * xScale; % L1 size
NYlayer = NY * yScale;

fprintf('scaled NX = %d scaled NY = %d\n',NXlayer,NYlayer);

read_data = 1;
PLOT_STEP = 10000;
skipRecords = 21400;
plot_rate_evol = 1;
plot_data = 0;

debug = 0;
scaleData = 1;
numRecords = 0;  % number of weights records (configurations)

if read_data
    %% open file pointers & figure handles
    
    fprintf('read data from file %s\n',filename);
    
    if exist(filename,'file')
        fid = fopen(filename, 'r', 'native');
    else
        disp(['Skipping, could not open ', filename]);
        return
    end
    
    %% read file header
    
    [time, NXlayer, NYlayer, NFlayer, numParams] =  readFirstHeader(fid);
    disp('read first header');
    fprintf('time = %f numParams = %d NX = %d NY = %d NF = %d\n',...
        time,numParams,NXlayer,NYlayer,NFlayer);
    disp('done');
    %pause
    N = NXlayer * NYlayer * NFlayer;
    
    % NOTE: The file may have the full header written before each record,
    % or only a time stamp
    
    k = 0;
    Data = [];
    numRecords = 0;
    
    while (~feof(fid) )
        
        % read the weights for this time step
        
        data = 1000.0 * fread(fid, N, 'float'); % N x 1 array
        
        if(~isempty(data))
            numRecords = numRecords + 1;
            if numRecords > skipRecords
                k=k+1;
                Data(k,:) =  data;
                
                if(k >=2 & plot_data)
                    data2D = reshape( data' - Data(k-1,:),[NX NY]);
                    imagesc(data2D,'CDataMapping','direct');
                    pause(0.1)
                end
                
                if (mod(k,PLOT_STEP) == 0)
                    %fprintf('time = %f\n',time);
                    
                    figure('Name',['Rate Field ' num2str(time)]);
                    data2D = reshape(data',[NX NY]);
                    imagesc(data2D,'CDataMapping','direct');
                    
                    %colorbar
                    axis square
                    axis off
                    
                    pause(0.1)
                    hold off
                end
                
            end % numRecords > skipRecords
            
        end % not empty data
        
        if ~feof(fid)
            numRecords = numRecords + 1;
            fprintf('numRecords = %d k = %d\n',numRecords, k);
            %pause
        end
    end % reading from data file
    
    
    fclose(fid);
    
    fprintf('feof reached: numRecords = %d \n',k);

end % read data


if plot_rate_evol
    disp('plot evolution of neuron rates');
    for k=1:size(Data,2)
        %fprintf('k=%d\n',k);
        plot(Data(:,k),'-b');
        if(k==1), hold on, end
        %pause
    end
end


% End primary function
%


function [time,NX,NY,NF,numParams] = readFirstHeader(fid)

% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int')

if head(3) ~= 1 % PVP_FILE_TYPE
    disp('incorrect file type')
    return
end
numParams = head(2)-2; % 18 params
fseek(fid,0,'bof'); % rewind file

params = fread(fid, numParams, 'int')
%pause
NX         = params(4);
NY         = params(5);
NF         = params(6);
fprintf('read header: numParams = %d ',numParams);
fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
% read time
time = fread(fid,1,'float64');
fprintf('time = %f, done!\n',time);

%pause
    
% End subfunction 
%
    
    
