function PATCH = stdp_plotPatchOld(fname, I, J)
% plot patch for neuron I, J 
global input_dir NK NO NX NY DTH 

filename = fname;
filename = [input_dir, filename];
N = NX * NY;
W_max = 10.0;

NXP = 3;
NYP = 3;
patch_size = NXP*NYP;

%figure;
%axis([0 NX*NXP  0 NY*NYP]);
fid=fopen(filename,'r');
%[H,count] = fscanf(fid, '%d %d %d %d %d')
%pause

W_array = fscanf(fid,'%g %g %g %g %g %g %g %g %g',[patch_size inf]);
W_array= W_array'; % this is (T*(NX*NY)) x patch_size
fclose(fid);
%W_array(1,:)
T=size(W_array,1)/(NX*NY);
%pause

W_max = 10;  
%colormap('gray');
colormap(jet);
b_color = 1;     % use to scale weights to the full range
                 % of the color map
a_color = (length(get(gcf,'Colormap'))-1.0)/W_max;

a= (NXP-1)/2;    % NXP = 2a+1;
b= (NYP-1)/2;    % NYP = 2b+1;


PATCH = [];

for t=1:T
    fprintf('t = %d\n',t);
    A = [];
    patch = W_array((t-1)*N + (J-1)*NX + I,:);
    A     = reshape(patch,[NXP NYP]);
    PATCH =  [PATCH; patch];
    
    if t==1
        Ainit = A;
        Aold = A;
    else
        
        imagesc(a_color*A+b_color);
        %imagesc(A-Ainit,'CDataMapping','direct');
        %imagesc(A,'CDataMapping','direct');
        %When CDataMapping is direct, the values of
        %CData should be in the range 1 to length(get(gcf,'Colormap'))
        %imagesc(A-Aold);
        pause(0.1)
        %Aold = A;
    end
end




