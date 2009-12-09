function A = stdp_compPCA(fname, xScale, yScale)
% plot last configuration of "weight" fields. 
% xScale and yScale are scale factors for this layer
% We should pass NX and NY as argumrnts

global input_dir % NX NY 

filename = fname;
filename = [input_dir, filename];
colormap(jet);
    
NX = 32;        % retina size
NY = 32;

NX = NX * xScale; % L1 size
NY = NY * yScale;

comp_evals  = 1;
write_spec = 1;
write_evecs = 1;
plot_evecs = 1;

evecs_file = [input_dir,'evecs.dat'];
cum_file   = [input_dir,'cum_evals.dat'];
evals_file = [input_dir,'evals.dat'];
 
figure('Name','Weights Fields');

debug = 0;


if exist(filename,'file')
    
    W_array = [];
    
    fid = fopen(filename, 'r', 'native');

    [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
        readFirstHeader(fid);
    fprintf('time = %f numPatches = %d NXP = %d NYP = %d NFP = %d\n',...
        time,numPatches,NXP,NYP,NFP);
    
    % Think of NXP and NYP as defining the size of the neuron's
    % receptive field that receives spikes from the retina.
    
    % read time
    time = fread(fid,1,'float64');
    fprintf('time = %f\n',time); 
        
    if numPatches ~= NX*NY
        disp('mismatch between numPatches and NX*NY')
        return
    end
    patch_size = NXP*NYP;
    
    [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP);
    fprintf('a = %d b = %d a1 = %d b1 = %d NXPbor = %d NYPbor = %d\n',...
        a,b,a1,b1,NXPbor,NYPbor);
                      
    PATCH = ones(NXPbor,NYPbor) * 122;
    
    %% read the last weights field (configuration)
    W_array = []; % N x patch_size array where N =NX * NY


    k=0;

    for j=1:NY
        for i=1:NX
            if ~feof(fid)
                k=k+1;
                nx = fread(fid, 1, 'uint16'); % unsigned short
                ny = fread(fid, 1, 'uint16'); % unsigned short
                nItems = nx*ny*NFP;
                if debug
                    fprintf('k = %d nx = %d ny = %d nItems = %d: ',...
                        k,nx,ny,nItems);
                end

                w = fread(fid, nItems, 'uchar'); % unsigned char
                % scale weights: they are quantized before written
                %w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
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


    % make the matrix of patches and plot patches for this time step
    A = [];

    if(~isempty(W_array))

        k=0;
        for j=(NYPbor/2):NYPbor:(NY*NYPbor)
            for i=(NXPbor/2):NXPbor:(NX*NXPbor)
                k=k+1;
                %W_array(k,:)
                patch = reshape(W_array(k,:),[NXP NYP]);
                PATCH(b1+1-b:b1+b,a1+1-a:a1+a) = patch';
%                 patch
%                 PATCH
%                 pause
                A(j-b1+1:j+b1,i-a1+1:i+a1) = PATCH;
                %imagesc(A,'CDataMapping','direct');
                %pause
            end
        end
        
        imagesc(A,'CDataMapping','direct');
        colorbar
        axis square
        axis off
        hold on


    end
    
    fclose(fid);

    %% compute principal weights components
    
    % compute covariance matrix
    
    C = cov(W_array);
    
    if(comp_evals)

        fprintf('compute evals\n')

        [V,D] = eig(C);      % note: KLvar = V D V^-1  

        % evecs stored in the columns of v
        % evals are on the diagonal of s

        evals = zeros(1,patch_size);

        for i=1:patch_size
            evals(patch_size+1-i) = D(i,i);
            %fprintf('e(%d)= %f\n',i,evals(patch_size+1-i))
        end

        figure('Name','Evals Spectrum');
        plot(evals,'-o')
        xlabel('i')
        ylabel('eval(i)')
        fprintf('type a char to continue:\n')
        pause


        % cummulative spectrum

        cum_evals = zeros(1,patch_size);

        for i=1:patch_size
            cum_evals(i) = sum(evals(1:i))/sum(evals) ;
        end

        figure('Name','Cumulative Spectrum');
        plot(cum_evals,'-o')
        xlabel('i')
        ylabel('cum_eval(i)')
        fprintf('type a char to continue:\n')
        pause

        % write evals and cum_evals

        if (write_spec)
            dlmwrite(cum_file,cum_evals',' ');
            dlmwrite(evals_file,evals',' ');
        end

    end % comp_evals


% plot evecs

if(plot_evecs)

  figure('Name','Evecs');

  for m=patch_size:-1:1
    norm(V(:,m),2);
    %V(:,m)
    patch = reshape(V(:,m),[NXP NYP]);
    evec = patch';
    imagesc(evec)
    fprintf('evec %d  (strike any key) \n',m);
    pause
  end
end  % end ploting evecs


% write evecs

if(write_evecs)
  
  fprintf('print evecs!\n')
  
  fid = fopen(evecs_file,'w');

  for v = patch_size:-1:1
      for j=1:patch_size
          fprintf(fid,'%12.8f ',V(j,v));
      end
      fprintf(fid,'\n');
  end

  fclose(fid);
end

    
    
else

    disp(['Skipping, could not open ', filename]);

end

% End primary function
%


function [time,numPatches,numParams,numWgtParams,NXP,NYP,NFP,minVal,maxVal] = ...
    readFirstHeader(fid)


% NOTE: see analysis/python/PVReadWeights.py for reading params
head = fread(fid,3,'int');
if head(3) ~= 3
    disp('incorrect file type')
    return
end
numWgtParams = 6;
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

wgtParams = fread(fid,numWgtParams,'int');
NXP = wgtParams(1);
NYP = wgtParams(2);
NFP = wgtParams(3);
minVal      = wgtParams(4);
maxVal      = wgtParams(5);
numPatches  = wgtParams(6);
fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
    minVal,maxVal,numPatches);
%pause

% End subfunction
%


function [time,varargout] = ...
    readHeader(fid,numParams,numWgtParams)

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

        wgtParams = fread(fid,numWgtParams,'int');
        NXP = wgtParams(1);
        NYP = wgtParams(2);
        NFP = wgtParams(3);
        minVal      = wgtParams(4);
        maxVal      = wgtParams(5);
        numPatches  = wgtParams(6);
        fprintf('NXP = %d NYP = %d NFP = %d ',NXP,NYP,NFP);
        fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
            minVal,maxVal,numPatches);

        varargout{1} = numPatches;
        varargout{2} = NXP;
        varargout{3} = NYP;
        varargout{4} = NFP;
        varargout{5} = minVal;
        varargout{6} = maxVal;
        %pause
    else
        disp('eof found: return');
        time = -1;
    end
else
    disp('eof found: return');
    time = -1;
end
% End subfunction
%



function [a,b,a1,b1,NXPbor,NYPbor] = compPatches(NXP,NYP)


if (mod(NXP,2) & mod(NYP,2))  % odd size patches
    a= (NXP-1)/2;    % NXP = 2a+1;
    b= (NYP-1)/2;    % NYP = 2b+1;
    NXPold = NXP;
    NYPold = NYP;
    NXPbor = NXP+2; % patch with borders
    NYPbor = NYP+2;
    a1= (NXPbor-1)/2;    % NXP = 2a+1;
    b1= (NYPbor-1)/2;    % NYP = 2b+1;

    dX = (NXPbor+1)/2;  % used in ploting the target
    dY = (NYPbor+1)/2;

else                 % even size patches

    a= NXP/2;    % NXP = 2a;
    b= NYP/2;    % NYP = 2b;
    NXPold = NXP;
    NYPold = NYP;
    NXPbor = NXP+2;   % add border pixels for visualization purposes
    NYPbor = NYP+2;
    a1=  NXPbor/2;    % NXP = 2a1;
    b1=  NYPbor/2;    % NYP = 2b1;

    dX = NXPbor/2;  % used in ploting the target
    dY = NYPbor/2;

end

% End subfunction
%