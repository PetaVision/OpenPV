function pvpdata = imagelisttopvp(infile, outfile)
% pvpdata = imagelisttopvp(infile, outfile)
%
% Reads in each of the images listed in the file specified by infile,
% and converts it to to a frame in a pvp activity file.
%
% The return value is a data structure of the same type as returned by
% readpvpfile when given a nonsparse activity pvp file.  The
% number of frames N is the same as the number of lines in infile, and
% the timestamps are 1, 2, ..., N.
%
% Hence, it can be used as input to writepvpactivityfile.
%
% If the second input argument is present, the pvpdata is saved to the
% path given by outfile.

fid = fopen(infile);
N = 0;
status = "";
while(~isequal(status, -1))
   status = fgets(fid, 256);
   if (ischar(status) && status(end)=="\n"), N = N+1; end
end%while

frewind(fid);

pvpdata = cell(N,1);

%[~,imagelist] = system(['cat ' infile]);
%eols = find(imagelist=="\n");
%linestarts = [1, eols(1:end-1)+1];
%linestops = eols-1;
%N = length(eols);
%
%pvpdata = cell(N,1);

for k=1:N
   fname = fgetl(fid);
   im = imread(fname);
   if isequal(class(im), 'logical'), conversionfactor = 1; end
   if isequal(class(im), 'uint8'), conversionfactor = 255; end
   if isequal(class(im), 'uint16'), conversionfactor = 65535; end
   im = double(im)/conversionfactor;
   im = permute(im, [2 1 3]);

   pvpdata{k}.time = k;
   pvpdata{k}.values = im;
end%for

fclose(fid); clear fid;

if nargin > 1
    writepvpactivityfile(outfile, pvpdata);
end
