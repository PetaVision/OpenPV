close all; clear all;
filename = '~/Documents/workspace/iHouse/LCAoutput/trace.txt';
spikeSc = 2;
in_file = fopen(filename, 'r');

if (in_file == -1)
   error('File doesnt exist');
end

trace = [];
spike = [];
while (true)
   string = fgetl(in_file);
   if (feof(in_file))
      break
   end
   [strSpike, strTrace] = strtok(string, ' ');
   spike = [spike; str2num(strSpike)];
   trace = [trace; str2num(strTrace)];
end
fclose(in_file);

figure;
plot(trace);
hold on
bar(spikeSc.*spike);
hold off


