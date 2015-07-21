function lc = linecount(filename)
%LINECOUNT Count the number of lines in given file
%
  %          LC = LINECOUNT(FILENAME) Returns LC, the number of
  %          lines in text file named FILENAME, or 0 if the file
%          does not exist or is not readable.

% Matthew Dailey 2000

  fid = fopen(filename,'r');
  if fid < 0
  lc = 0;
  else
    lc = 0;
while 1
ln = fgetl(fid);
if ~isstr(ln) break; end;
lc = lc + 1;
end;
fclose(fid);
end;
