function A = stdp_compScoreEvolution(fname, fname_last, numCenters, xScale, yScale)
% cluster the "weight" fields as a function of time
% and scores the learning of a set of pre-defined features
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as arguments
% NOTE: Needs to be implemented using ~/Documents/MATLAB/Kmeans


global input_dir  output_dir NX NY 

sym = {'or','ob','og','ok'};

NXscaled = NX * xScale; % L1 size
NYscaled = NY * yScale;

print_features = 0;
scaleWeights = 1;
write_centers =0;
write_scores = 1;
plot_centers = 1;
comp_kmeans = 1;
startTime = 0;
debug = 0;

if numel(fname) ~= numel(fname_last)
   disp('mismatch between weights files and last weights files: return');        
end


%% open file pointers & figure handles
for f=1:numel(fname)

    filename = fname{f};
    filename = [input_dir, filename];

    if exist(filename,'file')
        fid{f} = fopen(filename, 'r', 'native');
    else
        disp(['Skipping, could not open ', filename]);
        return
    end
    
    f_cluster(f) = figure('Name',['Layer ' num2str(f) ' Weights K-means Centers']);
end


%% define flags
first_record = ones(1,numel(fname));
last_record = zeros(1,numel(fname_last));


%% figure handle to score evolution plot

h_score = figure('Name','Learning Score Evolution');

%% open output file pointers

if(write_centers)
    centers_file = [output_dir,'WeightsKmeansCenters',num2str(numCenters),'.dat'];
    if exist(centers_file,'file')
        fid_centers = fopen(centers_file,'a');
        % read data, plot, and set startTime        
        startTime = 0;
    else
        fid_centers = fopen(centers_file,'w');
        startTime = 0;
    end
end

if(write_scores)
    scores_file = [output_dir,'WeightsLearningScores',num2str(numCenters),'.dat'];
    if exist(scores_file,'file')
        fid_scores = fopen(scores_file,'r');
        % read scores, plot, and set startTime
        data = fscanf(fid_scores, '%g %g %g', [3 inf]);   % It has two rows now.
        data = data';
        fclose(fid_scores);
        figure(h_score)
        plot(data(:,1),data(:,2),sym{1});hold on
        plot(data(:,1),data(:,3),sym{2});hold on
        
        %figure('Name','Score Correlation');
        %plot(data(:,2),data(:,3),'ob');
        %pause
        
        fid_scores = fopen(scores_file,'a');
        startTime = data(end,1) * 1000;
    else
        fid_scores = fopen(scores_file,'w');
        startTime = 0;
    end
end
%startTime = 3000000;
fprintf('startTime = %d \n',startTime);
pause
 
 
%% read first headers

for f=1:numel(fname)
    
    [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid{f});
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
        
    if numPatches ~= NXscaled*NYscaled
        disp('mismatch between numPatches and NX*NY: return');
        return
    end
    patch_size = NXP*NYP;
    
end
    
%% compute features matrix
Features = compFeatures(NXP, NYP);
% Features is a patch_size x numFeatures matrix
if print_features
    for f=1:size(Features,2)
        fprintf('feature %d: ',f);
        for i=1:patch_size
            fprintf('%.2f ',Features(i,f) );
        end
        fprintf('\n');
    end
    pause
end
        

%% read the weights field (configuration)
W_array = []; % N x patch_size array where N = NX * NY

eofFlag = 0;
    

%% loop over time
while (~eofFlag)    
    
    % loop over weights file
    for f=1:numel(fname)

        if ~first_record(f)
            [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                readHeader(fid{f},numParams);
            fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                time,numPatches,NXP,NYP,NFP);

            % detect last record and read from last configuration file
            if time == -1 & ~last_record(f)
                fclose(fid{f});
                last_record(f) = 1;

                filename = fname_last{f};
                filename = [input_dir, filename];

                if exist(filename,'file')
                    fid{f} = fopen(filename, 'r', 'native');
                else
                    disp(['Skipping, could not open ', filename]);
                    return
                end

                [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
                    readHeader(fid{f},numParams);
                fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
                    time,numPatches,NXP,NYP,NFP);
            end
                %pause
        else
            first_record(f) = 0;
        end
        
        k = 0;
        
        for j=1:NYscaled
        for i=1:NXscaled
            if ~feof(fid{f})
                k=k+1;
                nx = fread(fid{f}, 1, 'uint16'); % unsigned short
                ny = fread(fid{f}, 1, 'uint16'); % unsigned short
                nItems = nx*ny*NFP;
                if debug
                    fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                        k,nx,ny,nItems);
                end

                w = fread(fid{f}, nItems, 'uchar'); % unsigned char
                % scale weights: they are quantized before written
                if scaleWeights
                   w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                end
                if debug
                    for r=1:patch_size
                        fprintf('%d ',w(r));
                    end
                    fprintf('\n');
                    %pause
                end
                if(~isempty(w) & nItems ~= 0)
                    W_array(k,:) = w(1:patch_size);
                    %pause
                end
            end % if ~ feof
        end
    end % loop over post-synaptic neurons
    
   
    if comp_kmeans 

    %% compute K-means
    
    % data is n x patch_size where n = NX * NY
    [centers,mincenter,mindist,q2,quality] = kmeans(W_array,numCenters);
    %size(centers)  % numCenters x patch_size
    %size(mincenter)% n x 1
    %q2
    %quality

    % compute cluster weights
    for k=1:numCenters
        clustW(k) = sum(find(mincenter == k));
    end
    clustW = clustW ./sum(clustW);

    % normalize centers
    for k=1:numCenters
        centers(k,:) = centers(k,:) ./ norm(centers(k,:));
    end
    
    % sort centers according to weight
    
    [sortW,sortI] = sort(clustW,'descend');
    
    %% plot centers in reverse order of their weights
    numColumns = 4; % number of row plots
    numRows = ceil(numCenters/numColumns);
    
    if(plot_centers)
        %figure('Name',['Weights K-means Centers: time ' num2str(time)]);
        figure(f_cluster(f));
        for k=1:numCenters
            fprintf('cluster %d w = %f\n',k,sortW(k));
            %patch = clustW(k) * reshape(centers(k,:),[4 4])';
            patch = reshape(centers(sortI(k),:),[4 4])';
            subplot(numRows,numColumns,k);
            colormap gray
            imagesc(patch); % 'CDataMapping','direct'
            title(['Center ' num2str(k) ' w = ' num2str(clustW(sortI(k)),2) ] );
            %xlabel(['w = ' num2str(clustW(sortI(k)))] );
            colorbar
            axis square
            axis off
        end

    end

    
    %% write centers

    if(write_centers)
        if time >= 0
            if f==1,fprintf(fid_centers,'%d ',(startTime + time)/1000),end
            for k=1:numCenters
                fprintf(fid_centers,'%f ', sortW(k));
                for j=1:numel(centers(sortI(k),:))
                    fprintf(fid_centers,'%f ',centers(sortI(k),j));
                end
            end
        end
    end
    
    
    %% compute learning score
    learning_score = 0;
    for k=1:numCenters
        fprintf('cluster %d w = %f ',k,clustW(sortI(k)));
        [O, I ]= max(centers(sortI(k),:) * Features ) ;  
        fprintf('max_overlap = %f for feature %d\n', O, I); 
        learning_score = learning_score + sortW(k) * O;
    end
    fprintf('learning score = %f\n',learning_score);
    figure(h_score);
    if time >= 0
       plot([(startTime + time)/1000],[learning_score],sym{f});hold on
       fprintf('time = %d\n',(startTime + time)/1000 );
    end
    if time == 0 & f == 1
        xlabel('time');
        ylabel('learning score');
        pause
    end

    
    if(write_scores & time >= 0)
        if f==1,fprintf(fid_scores,'%d ',(startTime+time)/1000),end
        fprintf(fid_scores,'%f ',learning_score);        
    end
    
    end % comp_kmeans
    
    eofFlag = eofFlag | feof(fid{f});
  
end % loop over files

if(write_centers)
    fprintf(fid_centers,'\n');
end

if(write_scores)
    fprintf(fid_scores,'\n');
end
    

end % while loop over time

%% close files
if(write_centers),fclose(fid_centers),end
if(write_scores),fclose(fid_scores),end
for f=1:numel(fname)
   fclose(fid{f}); 
end

% End primary function
%


function [time,numPatches,numParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int');
if head(3) ~= 3
    disp('incorrect file type')
    return
end
numParams = head(2)-8;
fseek(fid,0,'bof'); % rewind file

params = fread(fid, numParams, 'int')
%pause
NX         = params(4);
NY         = params(5);
NF         = params(6);
fprintf('numParams = %d ',numParams);
fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
% read time
time = fread(fid,1,'float64');
fprintf('time = %f\n',time);

wgtParams = fread(fid,3,'int');
NXP = wgtParams(1);
NYP = wgtParams(2);
NFP = wgtParams(3);

rangeParams = fread(fid,2,'float');
minVal      = rangeParams(1);
maxVal      = rangeParams(2);
    
numPatches  = fread(fid,1,'int');


fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
    minVal,maxVal,numPatches);
%pause

% End subfunction
%


function [time,numPatches,NXP,NYP,NFP,minVal,maxVal] = ...
    readHeader(fid,numParams)

% NOTE: see analysis/python/PVReadWeights.py for reading params

if ~feof(fid)
    params = fread(fid, numParams, 'int')
    if numel(params)
        NXP         = params(4);
        NYP         = params(5);
        NFP         = params(6);
        fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
        % read time
        time = fread(fid,1,'float64');
        fprintf('time = %f\n',time);

        wgtParams = fread(fid,3,'int');
        NXP = wgtParams(1);
        NYP = wgtParams(2);
        NFP = wgtParams(3);
        
        
        rangeParams = fread(fid,2,'float');
        minVal      = rangeParams(1);
        maxVal      = rangeParams(2);
    
        numPatches  = fread(fid,1,'int');
    

        fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
        fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
            minVal,maxVal,numPatches);

        %varargout{1} = numPatches;
        %varargout{2} = NXP;
        %varargout{3} = NYP;
        %varargout{4} = NFP;
        %varargout{5} = minVal;
        %varargout{6} = maxVal;
        %pause
    else
        disp('eof found: return');
        time = -1;
        NXP = 0;
        NYP = 0;
        NFP = 0;
        minVal      = 0;
        maxVal      = 0;
        numPatches  = 0;
    end
else
    disp('eof found: return');
    time = -1;
    NXP = 0;
    NYP = 0;
    NFP = 0;
    minVal      = 0;
    maxVal      = 0;
    numPatches  = 0;
end
% End subfunction
%

function F = compFeatures(NXP, NYP)

% these are projections for 4x4 receptive fields
% in the Bars images experiments; it includes both
% vertical and horizontal features

F = zeros(NXP*NYP,8);
V = zeros(1,NXP*NYP);

% vertical features
for p=1:4
    V(:) = 0;
    ind = p:4:16;
    V(ind) = 1;
    V = V ./ norm(V);
    %patch = reshape(V,[NXP NYP])';
    %figure(p)
    %imagesc(patch,'CDataMapping','direct');
    F(:,p) = V'; % when V gets reshaped and transposed
                    % we get the right feature
    %pause
end

% horizontal features

for p=1:4
    V(:) = 0;
    V( (p-1)*4 +1 : p*4) = 1;
    V = V ./ norm(V);
    %patch = reshape(V,[NXP NYP])';
    %figure(p)
    %imagesc(patch,'CDataMapping','direct');
    F(:,4+p) = V'; % when V gets reshaped and transposed
                    % we get the right feature
    %pause
end


% End subfunction
%