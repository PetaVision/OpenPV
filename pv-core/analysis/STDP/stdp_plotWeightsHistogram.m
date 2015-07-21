function W_array = stdp_plotWeightsHistogram(fname)
% plot "weights" (typically after turning on just one neuron)
global input_dir n_time_steps NK NO NX NY DTH 

filename = fname;
filename = [input_dir, filename];

N=NX*NY;

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
    pause
    
    patch_size = NXP*NYP;
    %  weights
    T = 0;
    W_array = zeros(n_time_steps*N, patch_size);
    %W_array = [];
    
    k=0;
    while (~feof(fid))
        for j=1:NY
            for i=1:NX
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
                %fprintf('nxp = %d nyp = %d \n',nxp,nyp);
                %pause
                w = fread(fid, patch_size+3, 'uchar'); % unsigned char
                %pause
                % scale weights: they are quantized before are written
                w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                %if(~isempty(w))
                 %   W_array = [W_array; w(1:patch_size)']
                  %  pause
                %end
                if(~isempty(w))
                    k=k+1;
                    W_array(k,:) = w(1:patch_size);
                    %pause
                end
            end
        end
        if ~feof(fid)
            T = T + 1;
            fprintf('%d\n',T);
        end
    end
    fclose(fid);
    
    size(W_array)
    pause
    
    nbins = 100;
    
    % plot time dependent histograms
    n=0;
    for t=1:10:T
        fprintf('t = %d\n',t);
        A = W_array( ((t-1)*N + 1):(t*N), : ); % N x patch_size aray
        A = reshape(A, [1 (N*patch_size)] ) ;
        ind = find(A > 0.0);
        [n,xout] = hist(A(ind),nbins);
        plot(xout,n,'-r');
        hold on
        pause
        
    end


else
    
     disp(['Skipping, could not open ', filename]);
    
end
