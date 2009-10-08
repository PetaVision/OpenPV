function W_array = stdp_plotWeights(fname)
% plot "weights" (typically after turning on just one neuron)
global input_dir n_time_steps NK NO NX NY DTH 

filename = fname;
filename = [input_dir, filename];
%fprintf('NX = %d NY = %d\n',NX,NY);

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
    %pause
    
    patch_size = NXP*NYP;
    
    n_time_steps = 0;
    while (~feof(fid))
        for j=1:NY
            for i=1:NX
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
               % fprintf('nxp = %d nyp = %d : ',nxp,nyp);
                
                w = fread(fid, patch_size+3, 'uchar'); % unsigned char
                w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
%                 for k=1:patch_size
%                     fprintf('%f ',w(k));
%                 end
%                fprintf('\n');
%                pause
                %W_array = [W_array;w(1:patch_size)];
            end
        end
        n_time_steps = n_time_steps + 1;
        fprintf('%d\n',n_time_steps);
    end
    fclose(fid);
    fprintf('feof reached: n_time_steps = %d\n',n_time_steps);
    
else
    
     disp(['Skipping, could not open ', filename]);
    
end


T=size(W_array,1)/(NX*NY)
%pause
%W_max = 10;  
%colormap('gray');
colormap(jet);
b_color = 1;     % use to scale weights to the full range
                 % of the color map
a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

 
    
a= (NXP-1)/2;    % NXP = 2a+1;
b= (NYP-1)/2;    % NYP = 2b+1;


if 1 % version with  boundary around patch
    NXPold = NXP;
    NYPold = NYP;
    NXP = NXP+2;
    NYP = NYP+2;
    a1= (NXP-1)/2;    % NXP = 2a+1;
    b1= (NYP-1)/2;    % NYP = 2b+1;
    n=0;
    PATCH = zeros(NXP,NYP);
    %PATCH = repmat(...,NXP,NXP);
    
            
    for t=1:n_time_steps
        fprintf('t = %d\n',t);
        A = [];
        for i=((NXP+1)/2):NYP:(NY*NYP)
            for j=((NXP+1)/2):NXP:(NX*NXP)
                n=n+1;
                patch = reshape(W_array(n,:),[NXPold NYPold]);
                PATCH(a1+1-a:a1+1+a,b1+1-b:b1+1+b) = patch;
                A(i-a1:i+a1,j-b1:j+b1) = PATCH;
            end
        end
        
        if t==1
            Ainit = A;
            Aold = A;
        else
            
            %imagesc(A-Ainit,'CDataMapping','direct');
            %imagesc(A,'CDataMapping','direct');
            imagesc(a_color*A+b_color);
            axis square
            axis off
            %When CDataMapping is direct, the values of
            %CData should be in the range 1 to length(get(gcf,'Colormap'))
            %imagesc(A-Aold);
            pause(0.1)
            %Aold = A;
        end
    end
    
end % version with boundary around patch