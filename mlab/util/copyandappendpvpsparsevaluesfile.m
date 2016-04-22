function copyandappendpvpsparsevaluesfile(originalfilename, newfilename, newdata, nx, ny, nf)
   % copyandappendpvpsparsevaluesfile.m
   % -Wesley Chavez, 5/21/15
   % 
   % This script copies a pvp with sparse values, changes header.nbands, and appends it with sparse-valued newdata, in the same format returned by readpvpfile.
   % This is equivalent to, but is faster and uses less memory than:
   %
   % [originaldata header] = readpvpfile(originalfilename);
   % newdata = readpvpfile(anothersparsevaluedfilename);
   % appendeddata = [originaldata newdata];
   % writepvpsparsevaluesfile(newfilename, appendeddata, header.nx, header.ny, header.nf);
   
   if nargin ~= 6
       error('copyandappendpvpsparsevaluesfile:missingargs', 'copyandappendpvpsparsevaluesfile requires 6 arguments');
   end

   if ~ischar(originalfilename) || ~isvector(originalfilename) || size(originalfilename,1)~=1
       error('copyandappendpvpsparsevaluesfile:filenamenotstring', 'originalfilename must be a string');
   end

   if ~ischar(newfilename) || ~isvector(newfilename) || size(newfilename,1)~=1
       error('copyandappendpvpsparsevaluesfile:filenamenotstring', 'newfilename must be a string');
   end

   if ~iscell(newdata)
       error('copyandappendpvpsparsevaluesfile:newdatanotcell', 'newdata must be a cell array, one element per frame');
   end

   if ~isvector(newdata)
       error('copyandappendpvpsparsevaluesfile:newdatanotcellvector', 'newdata cell array must be a vector; either number of rows or number of columns must be one');
   end

   if isempty(newdata)
       error('copyandappendpvpsparsevaluesfile:newdataempty', 'newdata must have at least one frame');
   end

   numframes = length(newdata);
   numneurons = nx*ny*nf;

   for n=1:numframes
       if ~isempty(newdata{n}.values)
           if ismatrix(newdata{n}.values)
               sz = size(newdata{n}.values);
               if numel(sz)~=2 || sz(2)~=2
                   error('copyandappendpvpsparsevalues:badnumcols', 'newdata{%d}.values must have two columns', n);
               end
           else
               error('copyandappendpvpsparsevalues:nonmatrix', 'newdata{%d}.values must be a matrix with two columns', n);
           end
           if ~isnumeric(newdata{n}.values)
               error('copyandappendpvpsparsevalues:nonmatrix', 'newdata{%d}.values is not a numeric newdata type', n);
           end
           if ~isequal(newdata{n}.values(:,1), round(newdata{n}.values(:,1))) 
               error('copyandappendpvpsparsevalues:noninteger', 'newdata{%d}.values first column is not integral', n);
           end
           outofbounds = newdata{n}.values(:,1)<0 | newdata{n}.values(:,1)>=numneurons;
           if any(outofbounds)
                badindex = find(outofbounds, 1, 'first');
                badvalue = newdata{n}.values(badindex);
                error('copyandappendpvpsparsevalues:outofbounds', 'newdata{%d}.values first column must have values between 0 and nx*ny*nf-1=%d (first out-of-bounds value is entry (%d,1), value %d)', n, numneurons-1, badindex, badvalue);
           end
       end
   end
   
   errorpresent = 0;
   system(['cp ' originalfilename ' ' newfilename]); % Copy originalfilename, name it newfilename
   fid=fopen(newfilename,'rb+');
   fseek(fid,4*17,'bof'); % Fseek to header.nbands (number of data frames)
   originalNumBands = fread(fid,1,'int32');
   newNumBands = originalNumBands + length(newdata);
   fseek(fid,4*17,'bof'); 
   fwrite(fid,newNumBands,'int32'); % Overwrite with new nbands
   fseek(fid,0,'eof'); % Fseek to end of file and start writing newdata
   for frameno=1:length(newdata)   % allows either row vector or column vector.  isvector(newdata) was verified above
       fwrite(fid,newdata{frameno}.time,'double');
       count = size(newdata{frameno}.values,1);
       fwrite(fid,count,'uint32');
       for k=1:count
           fwrite(fid, newdata{frameno}.values(k,1),'uint32');
           fwrite(fid, newdata{frameno}.values(k,2),'single');
       end
   end

   fclose(fid); 
   clear fid;

   if errorpresent
       error(msgid, errmsg);
   end

end
