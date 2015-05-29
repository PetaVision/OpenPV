function resultTextFile(resultPvpFile, outputTextFilename, pv_dir, thresholdMin, thresholdMax, appendFlag)

if exist('pv_dir', 'var') && ~isempty(pv_dir)
   addpath(pv_dir);
end%if
if isempty(which('readpvpfile'))
   error("resultTextFile:readpvpfilemissing","resultTextFile error: missing command readpvpfile");
end%if

if ~exist('outputTextFilename','var') || isempty(outputTextFilename)
   fid =  1; % standard output
elseif exist('appendFlag','var') && appendFlag
   fid = fopen(outputTextFilename, "a");
else
   fid = fopen(outputTextFilename, "w");
end%if
if fid<0
   error("resultTextFile:cannotopen","resultTextFile error: unable to open \"%s\"", outputTextFilename);
end%if

classes={'background'; 'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; 'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; 'sofa'; 'train'; 'tvmonitor'};

A = readpvpfile(resultPvpFile);
for frame=1:numel(A)
   A1 = A{frame}.values;
   sz = size(A1);
   nx=sz(1);
   ny=sz(2);
   nf=sz(3);
   for y=1:ny
      for x=1:nx
         for f=1:nf
            conf=A1(x,y,f);
            if conf>=thresholdMin && conf<=thresholdMax
               fprintf(fid,"Image %3d, x=%d, y=%d, %-12s confidence=%f\n", frame, x, y, classes{f}, conf);
            end%if
         end%for
      end%for
   end%for
end%for

if fid>=3
   fclose(fid)
end%if
