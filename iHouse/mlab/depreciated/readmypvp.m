clear all; close all; more off;

global output_path;  output_path  = '/Users/slundquist/Documents/workspace/iHouse/output/';
global filename;     filename     = '/Users/slundquist/Documents/workspace/iHouse/output/retina.pvp';
global rootname;     rootname     = '00';

global OUT_FILE_EXT; OUT_FILE_EXT = 'png';             %either png or jpg for now

out = readpvpfile(filename);

[s, temp] = size(out);

for i=1:s
   max(out{i}.values)
   if(~isempty(out{i}.values))
      imwrite(out{i}.values, [output_path, num2str(i), '.jpg']);
   end
end
