function PATCH = stdp_plotPatch(fname, I, J, plot_title,fh)
% plot patch for neuron I, J 
global input_dir NK NO NX NY DTH 

if ~exist('fh','var')          % tests if 'fh' is a variable in the workspace
                               % returns 0 if 'fh' does not exists
    fh = figure('Name',plot_title);
else
    set(fh, 'Name', plot_title);
end

filename = fname;
filename = [input_dir, filename];
N = NX * NY;

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
    
patch_size = NXP*NYP;   


%colormap('gray');
colormap(jet);
b_color = 1;     % use to scale weights to the full range
                 % of the color map
a_color = (length(get(gcf,'Colormap'))-1.0)/maxVal;

a= (NXP-1)/2;    % NXP = 2a+1;
b= (NYP-1)/2;    % NYP = 2b+1;


PATCH = [];

nts = 0;

while (~feof(fid))
    for j=1:NY
        for i=1:NX
            
            nxp = fread(fid, 1, 'uint16'); % unsigned short
            nyp = fread(fid, 1, 'uint16'); % unsigned short
            w = fread(fid, patch_size+3, 'uchar'); % unsigned char

            if(i==I & j==J & ~isempty(w))
                %fprintf('nxp = %d nyp = %d : ',nxp,nyp);
                w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                A = reshape(w(1:patch_size), [NXP NYP]);
                PATCH =  [PATCH; w(1:patch_size)'];
            end
            
        end
    end
    nts = nts + 1;
    if (mod(nts,50) == 0)
        %imagesc(a_color*A+b_color);
        imagesc(A', 'CDataMapping','direct'); 
        % NOTE: It seems that I need A' here!!!
        colorbar
        pause(0.1)
        fprintf('%d\n',nts);
    end
end


