function [W, R] = stdp_analyzeWeightsRate (sumW, T, spike_array)
%UNTITLED Summary of this function goes here
%   sumW is an T x N array, where N=NX x NY.

global input_dir  NX NY write_step

plot_rate_weight = 0;

% spike bins
rate_bin_size = 100; 
tot_steps = size( spike_array, 1 );
rate_num_bins = fix( tot_steps / rate_bin_size );
rate_edges = 1:rate_bin_size:tot_steps;

% weight bins
weight_bin_size = rate_bin_size/write_step;
weight_num_bins = fix(T / weight_bin_size);
fprintf('eight_bin_size = %d weight_num_bins = %d', ...
    weight_bin_size, weight_num_bins);

N = size(sumW,2);
if N ~= (NX*NY)
   disp('wrong size for sumW')
   return
end
% NOTE: weight_num_bins and rate_num_bins must be the same

if weight_num_bins ~= rate_num_bins
   
    disp('mismatch between weight_num_bins and rate_num_bins')
    return
end

spike_rate = zeros(rate_num_bins*N,1);
average_weight = zeros(weight_num_bins*N,1);

for n=1:N
    
    fprintf('neuron %d\n',n);
    spike_time = find(spike_array(:,n));
    spike_rate(((n-1)*rate_num_bins + 1): n*rate_num_bins) = ...
        (1000/rate_bin_size) * histc(spike_time, rate_edges);
        % rate_num_bins x 1 array
  
    % mean returns a column of histogram counts
    average_weight(((n-1)*weight_num_bins + 1): n*weight_num_bins) = ...
        (mean( reshape(sumW(1:T,n), weight_bin_size,weight_num_bins),1))';
        % weight_num_bins x 1 array
    
    if plot_rate_weight
        plot(average_weight(((n-1)*N + 1): n*N),spike_rate(((n-1)*N + 1): n*N),'or');
        title(['Rate vs Weight for Neuron ' num2str(n)]);
        pause
    end
end



minW = min(average_weight);
maxW = max(average_weight);
hist(average_weight,100);

fprintf('minW = %f maxW = %f\n',minW, maxW);
bin_size = 1.0;
num_bins = round( (maxW-minW) / bin_size );
edges   = minW:bin_size:maxW;
[n,bin] = histc(average_weight,edges);
 % returns an index matrix in bin. If x is a vector, n(k) = sum(bin==k). 
 % bin is zero for out of range values. 
 file_name = 'AvRateAvWeight.txt';
 fid = fopen(file_name,'w');
 
 for k=1:num_bins
     
    W(k) = minW + (k-0.5)* bin_size ;
    ind = find(bin==k);
    R(k) = mean(spike_rate(ind));
     fprintf(fid,'%f %f\n',W(k),R(k));
 end
 fclose(fid);
 
figure('Name','Rate vs Weight');
plot(W,R,'ob');
xlabel(texlabel('< Sigma_j w_{ij} >_t'));
ylabel('< R(i) >_t');