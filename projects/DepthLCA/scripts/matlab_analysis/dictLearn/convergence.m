basedirs = { ...
   '~/mountData/dictLearn/convergence/icapatch_binoc/'; ...
%   '~/mountData/dictLearn/convergence/icapatch_mono/'; ...
   };
labels = { ...
   'Binocular'; ...
   'Monocular';
};
color = { ...
   'r'; ...
   'b'; ...
};
outFilename = '~/mountData/dictLearn/convergence/outplots/binoc_convergence.png'

%Grab lines 2000, 3000, 4000, and 5000
linenums = [2000, 3000, 4000, 5000];

fig = figure;
hold on;

for i = 1:length(basedirs)
   basedir = basedirs{i};
   %Get directories in dir
   dirs = dir(basedir);
   %Remove . and ..
   dirs(1) = [];
   dirs(1) = [];

   dirNames = {dirs.name};
   numPoints = length(dirNames);

   timestep = zeros(numPoints, 1);
   energy = zeros(numPoints, 1);

   for di = 1:numPoints
      timestep(di, 1) = str2num(strsplit(dirNames{di}, '_'){end});
      energyFile = [basedir, dirNames{di}, '/total_energy.txt'];

      [fid, errmessage] = fopen(energyFile, 'r');
      if(fid < 0) 
         energyFile
         disp(errmessage)
         keyboard
      end
      tempEnergy = zeros(length(linenums), 1);
      for(li = 1:length(linenums))
         linenum = linenums(li);
         line = textscan(fid, '%s', 1, 'delimiter', '\n', 'headerlines', linenum-1){1}{1};
         fseek(fid, 0, 'bof');
         tempEnergy(li) = str2num(strsplit(line, ','){end});
      end
      fclose(fid);
      energy(di) = mean(tempEnergy);
   end

   %Sort by timestep
   [timestep, idx] = sort(timestep);
   energy = energy(idx);

   plot(timestep, energy, color{i}, 'lineWidth', 5);
end

hold off;

title('Energy vs Time', 'FontSize', 28);
xlabel('Timestep', 'FontSize', 28);
ylabel('Energy', 'FontSize', 28);

if(length(basedirs) > 1)
   h = legend(labels);
   set(h, 'FontSize', 16);
end

saveas(
fig, outFilename);
