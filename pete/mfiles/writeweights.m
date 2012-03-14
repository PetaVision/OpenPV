function writeweights(w,filename,m,n,f,timestamp)
% writeweights(w,filename,m,n,f,timestamp)
% w is a 2-D array where w(i,j) is the weight of the i^th feature
% of the j^th patch.  For the time being, nxp and nyp must equal 1.
% filename is the path to the filename.  Any existing file named filename
% is clobbered.
% m,n,f are the dimensions of the presynaptic layer (stored in index_nx
% and index_ny of the header)
% timestamp is the time to store in the index_time field of the header

shortintsize = 2;
floatsize = 4;
[numfeatures, numpatches] = size(w);
recordsize = numpatches*(floatsize*numfeatures+2*shortintsize);
header = zeros(18,1);
header(1) = 104;                % index_header_size
header(2) = 26;                 % index_num_params
header(3) = 5;                  % index_file_type
header(4) = m;                  % index_nx
header(5) = n;                  % index_ny
header(6) = f;                  % index_nf
header(7) = 1;                  % index_num_records
header(8) = recordsize;         % index_record_size
header(9) = 4;                  % index_data_size
header(10) = 3;                 % index_data_type
header(11) = 1;                 % index_nx_procs
header(12) = 1;                 % index_ny_procs
header(13) = m;                 % index_nx_global
header(14) = n;                 % index_ny_global
header(15) = 0;                 % index_kx0
header(16) = 0;                 % index_ky0
header(17) = 0;                 % index_nb
header(18) = 1;                 % index_nbands
% timestamp                     % index_time
patchsize = zeros(3,1);
patchsize(1) = 1;               % index_wgt_nxp
patchsize(2) = 1;               % index_wgt_nyp
patchsize(3) = numfeatures;     % index_wgt_nfp
wgtinterval = zeros(2,1);
wgtinterval(1) = min(w(:));     % index_wgt_min
wgtinterval(2) = max(w(:));     % index_wgt_max
%numpatches = numpatches;       % index_wgt_numpatches

fid = fopen(filename,'w');
fwrite(fid,header,'int32');
fwrite(fid,timestamp,'float64');
fwrite(fid,patchsize,'int32');
fwrite(fid,wgtinterval,'float32');
fwrite(fid,numpatches,'int32');
for pixel=1:numpatches
    fwrite(fid,[1 1],'int16');
    fwrite(fid,w(:,pixel),'float32');
end

fclose(fid); clear fid;