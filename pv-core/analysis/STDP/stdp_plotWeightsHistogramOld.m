function W_array = stdp_plotWeightsHistogram(fname)
% plot "weights" (typically after turning on just one neuron)
global input_dir NK NO NX NY DTH 

filename = fname;
filename = [input_dir, filename];

NXP = 3;
NYP = 3;
patch_size = NXP*NYP;

if exist(filename,'file')
    
    fid=fopen(filename,'r');
    %[H,count] = fscanf(fid, '%d %d %d %d %d')
    %pause
    
    W_array = fscanf(fid,'%g %g %g %g %g %g %g %g %g',[patch_size inf]);
    W_array= W_array';
    fclose(fid);
    %W_array(1,:)
    T=size(W_array,1)/(NX*NY);
    
    %size(W_array)
    %pause
    
    
    N=NX*NY;
    nbins = 50;
    
    n=0;
    for t=1:T
        fprintf('t = %d\n',t);
        A = W_array( ((t-1)*N + 1):(t*N), : ); % N x patch_size aray
        A = reshape(A, [1 (N*patch_size)] ) ;
        ind = find(A > 0.0);
        [n,xout] = hist(A(ind),nbins);
        
        plot(xout,n,'-r');
        %hold on
        pause(0.1)
        
    end


else
    
     disp(['Skipping, could not open ', filename]);
    
end
