close all;

dir = '/Users/slundquist/Documents/workspace/BIDS/output/csv/';
node1BS_fn = [dir, 'BS_117_124.csv']; 
node1AS_fn = [dir, 'AS_117_124.csv']; 
node2BS_fn = [dir, 'BS_32_111.csv']; 
node2AS_fn = [dir, 'AS_32_111.csv']; 

node1BS = csvread(node1BS_fn);
node1AS = csvread(node1AS_fn);
%node2BS = csvread(node2BS_fn);
%node2AS = csvread(node2AS_fn);

node1BS_data = node1BS(:,1);
node1BS_sin = node1BS(:,2);
node1BS_cos= node1BS(:,3);
node1BS_dsin = node1BS(:,4);
node1BS_dcos= node1BS(:,5);
node1BS_hsin = node1BS(:,6);
node1BS_hcos= node1BS(:,7);

node1AS_data = node1AS(:,1);
node1AS_sin = node1AS(:,2);
node1AS_cos= node1AS(:,3);
node1AS_dsin = node1AS(:,4);
node1AS_dcos= node1AS(:,5);
node1AS_hsin = node1AS(:,6);
node1AS_hcos= node1AS(:,7);

if(node1BS_sin ~= node1AS_sin)
   die('shit aint equal');
end

if(node1BS_cos~= node1AS_cos)
   die('shit aint equal');
end

if(node1BS_dsin ~= node1AS_dsin)
   die('shit aint equal');
end

if(node1BS_dcos~= node1AS_dcos)
   die('shit aint equal');
end

if(node1BS_hsin ~= node1AS_hsin)
   die('shit aint equal');
end

if(node1BS_hcos~= node1AS_hcos)
   die('shit aint equal');
end

figure;
plot(node1BS_data);
title('BS data')

figure;
plot(node1AS_data);
title('AS data')

figure;
plot(node1BS_sin);
title('sin')

figure;
plot(node1BS_cos);
title('cos')

figure;
plot(node1BS_dcos);
title('double cos');

figure;
plot(node1BS_dsin);
title('double sin');

figure;
plot(node1BS_hcos);
title('half cos');

figure;
plot(node1BS_hsin);
title('half sin');

node1BS_sin_sum = sum(node1BS_data .* node1BS_sin)
node1BS_cos_sum = sum(node1BS_data .* node1BS_cos)

node1BS_dsin_sum = sum(node1BS_data .* node1BS_dsin)
node1BS_dcos_sum = sum(node1BS_data .* node1BS_dcos)

node1BS_hsin_sum = sum(node1BS_data .* node1BS_hsin)
node1BS_hcos_sum = sum(node1BS_data .* node1BS_hcos)

node1AS_sin_sum = sum(node1AS_data .* node1AS_sin)
node1AS_cos_sum = sum(node1AS_data .* node1AS_cos)

node1AS_dsin_sum = sum(node1AS_data .* node1AS_dsin)
node1AS_dcos_sum = sum(node1AS_data .* node1AS_dcos)

node1AS_hsin_sum = sum(node1AS_data .* node1AS_hsin)
node1AS_hcos_sum = sum(node1AS_data .* node1AS_hcos)

node1BS_C  = sqrt(node1BS_sin_sum^2 + node1BS_sin_sum^2)
node1BS_dC = sqrt(node1BS_dsin_sum^2 + node1BS_dsin_sum^2)
node1BS_hC = sqrt(node1BS_hsin_sum^2 + node1BS_hsin_sum^2)

node1AS_C  = sqrt(node1AS_sin_sum^2 + node1AS_sin_sum^2)
node1AS_dC = sqrt(node1AS_dsin_sum^2 + node1AS_dsin_sum^2)
node1AS_hC = sqrt(node1AS_hsin_sum^2 + node1AS_hsin_sum^2)

node1BS_norm_val = (node1BS_dC + node1BS_hC)/2
node1AS_norm_val = (node1AS_dC + node1AS_hC)/2

node1BS_out = log(node1BS_C/node1BS_norm_val)/log(2)
node1AS_out = log(node1AS_C/node1AS_norm_val)/log(2)
