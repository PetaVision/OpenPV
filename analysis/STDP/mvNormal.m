
% gen multivariate normal distributions with different centers and 
% covariance matrices for G-means analysis 


out_dir = '/Users/manghel/Documents/MATLAB/Xmeans/';

filename = [out_dir '5class.ds' ];
fid = fopen(filename, 'w');
fprintf(fid,'# num_rows = 2000\n');
fprintf(fid,'# num_cols = 2\n');
fprintf(fid,'# num_classes = 5\n');
fprintf(fid,'\n\nx0 x1\n');

% set 1
mu = [2 3];
SIGMA = [1 1.5; 1.5 3];
r = mvnrnd(mu,SIGMA,400); % 200 x 2 array
plot(r(:,1),r(:,2),'or');
hold on
for i=1:length(r)
   fprintf(fid,'%15.13f %15.13f\n',r(i,1),r(i,2)) ;
end


% set 2
mu = [-8 8];
SIGMA = [1 0; 0 3];
r = mvnrnd(mu,SIGMA,400);
plot(r(:,1),r(:,2),'ob');
for i=1:length(r)
   fprintf(fid,'%15.13f %15.13f\n',r(i,1),r(i,2)) ;
end


% set 3
mu = [8 -8];
SIGMA = [3 1.5; 1.5 1];
r = mvnrnd(mu,SIGMA,400);
plot(r(:,1),r(:,2),'og');
hold on
for i=1:length(r)
   fprintf(fid,'%15.13f %15.13f\n',r(i,1),r(i,2)) ;
end


% set 4
mu = [-8 -8];
SIGMA = [1 0; 0 1];
r = mvnrnd(mu,SIGMA,400);
plot(r(:,1),r(:,2),'ok');
for i=1:length(r)
   fprintf(fid,'%15.13f %15.13f\n',r(i,1),r(i,2)) ;
end

% set 5
mu = [2 -16];
SIGMA = [3 1.5; 1.5 1];
r = mvnrnd(mu,SIGMA,400);
plot(r(:,1),r(:,2),'om');
hold on
for i=1:length(r)
   fprintf(fid,'%15.13f %15.13f\n',r(i,1),r(i,2)) ;
end


fclose(fid)