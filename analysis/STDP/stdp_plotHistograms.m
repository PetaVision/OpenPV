% plots synaptoc weights histograms for a range of
% retina firing rates.
% NOTE: The weights are quantized in the [0,255] range.

input_dir = '/Users/manghel/Documents/workspace/marian/output/';
fname = 'w0_last_hist_';

minVal = 0;
maxVal = 3; % STDP values

x = 0:1:255;
x =  minVal + (maxVal - minVal) * ( x * 1.0)/ 255.0;
C = get(gca,'ColorOrder'); % Nx3 matrix
N = size(C,1);
p = 0;              % plot index
for f =10:10:100 % retina firing rate
    filename = [input_dir, fname, num2str(f), '.dat'];
    fid=fopen(filename,'r');
    h=fscanf(fid,'%d',inf);
    
        plot(x,h,'o-','MarkerFaceColor',C(mod(p,N)+1,:));hold on
    
    p=p+1;
    %pause
end
hold off

