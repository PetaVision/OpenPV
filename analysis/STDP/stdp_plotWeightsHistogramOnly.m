function A = stdp_plotWeightsHistogramOnly(fname, nbins, TSTEP)
% At each time step plots the weights histopgram.
% Returns the last weights distribution.

global input_dir n_time_steps NK NO NX NY 

filename = fname;
filename = [input_dir, filename];

N=NX*NY;
bufSize = 4; % see pv_write_patch() in io.c
nPad = 1;    % size of the layer padding
nf = 1;      % number of features

if exist(filename,'file')
    
    fid=fopen(filename,'r','native');
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
    %pause
    
    patch_size = NXP*NYP;
    
    T = 0;
    
    % Think of nx and ny as defining the size of the neuron's
    % receptive field that receives spikes from the retina.
    while (~feof(fid))
        W_array = []; % reset every time step: this is N x patch_size array
        % where N =NX x NY
        
        % read boundary neurons
        for i=1:(NX+2*nPad)
            nx = fread(fid, 1, 'uint16'); % unsigned short
            ny = fread(fid, 1, 'uint16'); % unsigned short
            nItems = nx*ny*nf;
            fprintf('nx = %d ny = %d \n',nx,ny);
            %pause

            w = fread(fid, nItems, 'uchar'); % unsigned char

        end
        pause
                
        k = 0;
        for j=1:NY
            for i=1:NX
                nx = fread(fid, 1, 'uint16'); % unsigned short
                ny = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nx*ny*nf;
                fprintf('nx = %d ny = %d \n',nx,ny);
                %pause
                if nItems <= 4
                w = fread(fid, 4, 'uchar'); % unsigned char
                else if nItmes <= 8
                %pause
                % scale weights: they are quantized before are written
                w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                
                if(~isempty(w))
                    k=k+1;
                    W_array(k,:) = w(1:patch_size);
                    %pause
                end
            end
        end
        if ~feof(fid)
            T = T + 1;
            %fprintf('%d\n',T);
        end
        
        % plot the weights histogram for the first time step
        
        if ( T == 1 && ~isempty(W_array) )
            [m,n]=size(W_array);
            fprintf('%d %d %d\n',T,m,n);
            A = reshape(W_array, [1 (N*patch_size)] ) ;
            %ind = find(A > 0.0);
            %[n,xout] = hist(A(ind),nbins);
            [n,xout] = hist(A,nbins);
            %plot(xout,n,'-g','LineWidth',3);
            figure('Name', ['Time ' num2str(T) ' Weights Histogram']);
            %bar(xout,n);
            hist(A,nbins);
            %hold on
            pause
            
        end
        
        
        % plot the weights histogram for this time step
        
        if ( ~mod(T,TSTEP) && ~isempty(W_array) )
            [m,n]=size(W_array);
            fprintf('%d %d %d \n',T,m,n);
            A = reshape(W_array, [1 (N*patch_size)] ) ;
            %ind = find(A > 0.0);
            %[n,xout] = hist(A(ind),nbins);
            [n,xout] = hist(A,nbins);
            %plot(xout,n,'-r');
            figure('Name', ['Time ' num2str(T) ' Weights Histogram']);
            bar(xout,n,'r');
            %hold on
            pause(0.1)
            
        end
        
    end
    fclose(fid);
    plot(xout,n,'-b','LineWidth',3);
    


else
    
     disp(['Skipping, could not open ', filename]);
    
end
