function A = stdp_plotWeightsOnly(fname, Xtarg, Ytarg)
% plot "weights" (typically after turning on just one neuron)
% Xtarg and Ytarg contain the X and Y coordinates of the target
global input_dir n_time_steps % NX NY 

filename = fname;
filename = [input_dir, filename];
%fprintf('NX = %d NY = %d\n',NX,NY);
colormap(jet);
    
NX = 32;
NY = 32;
xScale = 2;
yScale = 2;
NX = NX * xScale;
NY = NY * yScale;

PLOT_STEP = 5;
plotTarget = 0;

figure('Name','Weights Fields');

debug = 0;
bufSize = 4; % see pv_write_patch() in io.c
nPad = 2;    % size of the layer padding
nf = 1;      % number of features

if exist(filename,'file')
    
    W_array = [];
    
    fid = fopen(filename, 'r', 'native');
    %     header
    %     params[0] = header_size
    %     params[1] = nParams;
    %     params[2] = file_type
    %     params[3] = nxp;
    %     params[4] = nyp;
    %     params[5] = nfp;
    %     params[6] = (int) minVal;        // stdp value
    %     params[7] = (int) ceilf(maxVal); // stdp value
    %     params[8] = numPatches;
    %
    header_size = fread(fid, 1, 'int');
    num_params  = fread(fid, 1, 'int');
    file_type   = fread(fid, 1, 'int');   
    NXP         = fread(fid, 1, 'int');
    NYP         = fread(fid, 1, 'int');
    NFP         = fread(fid, 1, 'int');
    minVal      = fread(fid, 1, 'int');
    maxVal      = fread(fid, 1, 'int');
    numPatches = fread(fid, 1, 'int');
    fprintf('header_size = %d num_params = %d file_type = %d \n',...
        header_size,num_params,file_type);
    fprintf('NXP = %d NYP = %d NFP = %d ',...
       NXP,NYP,NFP);
    fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
        minVal,maxVal,numPatches);
    %pause
    if numPatches ~= NX*NY
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    if (mod(NXP,2) & mod(NYP,2))  % odd size patches
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
        
    else                 % even size patches
        
        a= NXP/2;    % NXP = 2a;
        b= NYP/2;    % NYP = 2b;
        NXPold = NXP;
        NYPold = NYP;
        NXP = NXP+2;   % add border pixels for visualization purposes
        NYP = NYP+2;
        a1=  NXP/2;    % NXP = 2a1;
        b1=  NYP/2;    % NYP = 2b1;

        dX = NXP/2;  % used in ploting the target
        dY = NYP/2;



    end
    
    b_color = 1;     % use to scale weights to the full range
                 % of the color map
    a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

    n_time_steps = 0;
    %W_array = zeros(NX*NY,patch_size);
    %PATCH = ones(NXP,NYP) * (length(get(gcf,'Colormap'))/2);
    PATCH = ones(NXP,NYP) * (0.5*(maxVal+minVal));
    
    avWeights = [];  % time averaged weights array
    
     % Think of nx and ny as defining the size of the neuron's
    % receptive field that receives spikes from the retina.
    while (~feof(fid))
        
        % read the weights for this time step 
        W_array = []; % reset every time step: this is N x patch_size array
                      % where N =NX * NY
                      
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
                if ~feof(fid)
                    k=k+1;
                    nx = fread(fid, 1, 'uint16'); % unsigned short
                    ny = fread(fid, 1, 'uint16'); % unsigned short
                    nItems = nx*ny*nf;
                    if mod(nItems,bufSize)
                        nRead = (floor(nItems/bufSize) + 1)*bufSize;
                    else
                        nRead = floor(nItems/bufSize) * bufSize;
                    end
                    if debug & n_time_steps >= 0
                        fprintf('k = %d nx = %d ny = %d nItems = %d nRead = %d: ',...
                            k,nx,ny,nItems,nRead);
                    end
                    if nRead~= 0
                        w = fread(fid, nRead, 'uchar'); % unsigned char
                        % scale weights: they are quantized before written
                        w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                    end
                    if debug & n_time_steps >= 0
                        for r=1:patch_size
                            fprintf('%f ',w(r));
                        end
                        fprintf('\n');
                        pause
                    end
                    if(~isempty(w) & nRead ~= 0)
                        W_array(k,:) = w(1:patch_size);
                        %pause
                    end
                end % if ~ feof 
            end
        end % loop over post-synaptic neurons
        if ~feof(fid)
            n_time_steps = n_time_steps + 1;
            fprintf('k = %d time = %d\n',k,n_time_steps);
        end
        
        
        
        % make the matrix of patches and plot patches for this time step
        A = [];
        
        if(~isempty(W_array))
            
            
            k=0;
            for j=(NYP/2):NYP:(NY*NYP)
                for i=(NXP/2):NXP:(NX*NXP)
                    k=k+1;
                    %W_array(k,:)
                    patch = reshape(W_array(k,:),[NXPold NYPold]);
                    PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
                    %patch'
                    %pause
                    A(j-b1+1:j+b1,i-a1+1:i+a1) = PATCH;
                    %imagesc(A,'CDataMapping','direct');
                    %pause
                end
            end
            
            if n_time_steps==1
                Ainit = A;
                Aold = A;
                avWeights = A;
                fprintf('time = %d\n',n_time_steps);
                imagesc(A,'CDataMapping','direct');
                colorbar
                axis square
                axis off
                hold on
            else
                %imagesc(a_color*A+b_color);
                %imagesc(A-Ainit,'CDataMapping','direct');
                if (mod(n_time_steps,PLOT_STEP) == 0)
                    fprintf('time = %d\n',n_time_steps);
                    imagesc(A,'CDataMapping','direct');
                    colorbar
                    axis square
                    axis off
                    hold on
                    % plot target pixels
                    if plotTarget
                        for t=1:length(Xtarg)
                            I=Xtarg(t);
                            J=Ytarg(t);
                            plot([(J-1)*NXP+dY],[(I-1)*NXP+dX],'.r','MarkerSize',12)
                        end
                    end
                    pause(0.1)
                    hold off
                end
                Aold = A;
                avWeights = avWeights + A;
            end
        end
        %pause
    end % reading from weights file
    fclose(fid);
    fprintf('feof reached: n_time_steps = %d\n',n_time_steps);
    avWeights = avWeights / n_time_steps;
    figure('Name','Time Averaged Weights');
    imagesc(avWeights,'CDataMapping','direct');
    colorbar
    axis square
    axis off
else
    
     disp(['Skipping, could not open ', filename]);
    
end


