function stdp_plotWeightsOnly(fname, Xtarg, Ytarg)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
global input_dir n_time_steps % NX NY 

filename = fname;
filename = [input_dir, filename];
%fprintf('NX = %d NY = %d\n',NX,NY);
colormap(jet);
    
NX = 32;NY=32;
PLOT_STEP = 50;

figure('Name','Weights Fields');

debug = 0;
bufSize = 4; % see pv_write_patch() in io.c
nPad = 1;    % size of the layer padding
nf = 1;      % number of features

if exist(filename,'file')
    
    W_array = [];
    
    fid = fopen(filename, 'r', 'native');
    %     header
    %     params[0] = nParams;
    %     params[1] = nxp;
    %     params[2] = nyp;
    %     params[3] = nfp;
    %     params[4] = (int) minVal;        // stdp value
    %     params[5] = (int) ceilf(maxVal); // stdp value
    %     params[6] = numPatches;
    %
    num_params = fread(fid, 1, 'int');
    NXP = fread(fid, 1, 'int');
    NYP = fread(fid, 1, 'int');
    NFP = fread(fid, 1, 'int');
    minVal = fread(fid, 1, 'int');
    maxVal = fread(fid, 1, 'int');
    numPatches = fread(fid, 1, 'int');
    fprintf('num_params = %d NXP = %d NYP = %d NFP = %d ',...
        num_params,NXP,NYP,NFP);
    fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
        minVal,maxVal,numPatches);
    pause
    
    patch_size = NXP*NYP;
    
    a= (NXP-1)/2;    % NXP = 2a+1;
    b= (NYP-1)/2;    % NYP = 2b+1;
    NXPold = NXP;
    NYPold = NYP;
    NXP = NXP+2;
    NYP = NYP+2;
    a1= (NXP-1)/2;    % NXP = 2a+1;
    b1= (NYP-1)/2;    % NYP = 2b+1;
    
    dX = (NXP+1)/2;  % used in ploting the target
    dY = (NYP+1)/2;
    
    b_color = 1;     % use to scale weights to the full range
                 % of the color map
    a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

    n_time_steps = 0;
    %W_array = zeros(NX*NY,patch_size);
    %PATCH = ones(NXP,NYP) * (length(get(gcf,'Colormap'))/2);
    PATCH = ones(NXP,NYP) * maxVal;
    
    avWeights = [];  % time averaged weights array
    
     % Think of nx and ny as defining the size of the neuron's
    % receptive field that receives spikes from the retina.
    while (~feof(fid))
        
        % read the weights for this time step 
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX x NY
                      
        % read boundary neurons
%         for i=1:(NX+2*nPad)
%             nx = fread(fid, 1, 'uint16'); % unsigned short
%             ny = fread(fid, 1, 'uint16'); % unsigned short
%             nItems = nx*ny*nf;
%             fprintf('nx = %d ny = %d ',nx,ny);
%             %pause
%             if mod(nItems,bufSize)
%                 nRead = (floor(nItems/bufSize) + 1)*bufSize;
%             else
%                 nRead = floor(nItems/bufSize) * bufSize;
%             end
%             fprintf(' nRead = %d\n',nRead);
%             w = fread(fid, nRead, 'uchar'); % unsigned char
% 
%         end
%         pause              
                      
        k=0;
        
        for j=1:NY
            for i=1:NX
                k=k+1;
                nx = fread(fid, 1, 'uint16'); % unsigned short
                ny = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nx*ny*nf;
                if mod(nItems,bufSize)
                    nRead = (floor(nItems/bufSize) + 1)*bufSize;
                else
                    nRead = floor(nItems/bufSize) * bufSize;
                end
                if debug
                    fprintf('nx = %d ny = %d nRead = %d: ',nx,ny,nRead);
                end
                if nRead~= 0
                    w = fread(fid, nRead, 'uchar'); % unsigned char
                    % scale weights: they are quantized before are written
                    w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                end
                if debug
                    for r=1:patch_size
                        fprintf('%f ',w(r));
                    end
                    fprintf('\n');
                    %pause
                end
                if(~isempty(w) & nRead ~= 0)
                    W_array(k,:) = w(1:patch_size);
                    %pause
                end
            end
        end
        if ~feof(fid)
            n_time_steps = n_time_steps + 1;
            %fprintf('%d\n',n_time_steps);
        end
        
        
        
        % make the matrix of patches and plot patches for this time step
        A = [];
        
        if(~isempty(W_array))
            
            
            k=0;
            for j=((NYP+1)/2):NYP:(NY*NYP)
                for i=((NXP+1)/2):NXP:(NX*NXP)
                    
                    k=k+1;
                    patch = reshape(W_array(k,:),[NXPold NYPold]);
                    PATCH(a1+1-a:a1+1+a,b1+1-b:b1+1+b) = patch;
                    %pause
                    A(i-a1:i+a1,j-b1:j+b1) = PATCH;
                    %imagesc(A,'CDataMapping','direct');
                    %pause
                end
            end
            
            if n_time_steps==1
                Ainit = A;
                Aold = A;
                avWeights = A;
            else
                %imagesc(a_color*A+b_color);
                %imagesc(A-Ainit,'CDataMapping','direct');
                if (mod(n_time_steps,PLOT_STEP) == 0)
                    fprintf('%d\n',n_time_steps);
                    imagesc(A','CDataMapping','direct');
                    colorbar
                    axis square
                    axis off
                    hold on
                    % plot target pixels
                    for t=1:length(Xtarg)
                        I=Xtarg(t);
                        J=Ytarg(t);
                        plot([(I-1)*NXP+dX],[(J-1)*NXP+dY],'.r','MarkerSize',12)
                    end
                    pause(0.1)
                    hold off
                end
                Aold = A;
                avWeights = avWeights + A;
            end
        end
    end % reading from weights file
    fclose(fid);
    fprintf('feof reached: n_time_steps = %d\n',n_time_steps);
    avWeights = avWeights / n_time_steps;
    figure('Name','Time Averaged Weights');
    imagesc(avWeights','CDataMapping','direct');
    colorbar
    axis square
    axis off
else
    
     disp(['Skipping, could not open ', filename]);
    
end


