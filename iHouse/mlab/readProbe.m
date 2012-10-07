%filename = '~/Documents/workspace/iHouse/output/LCALIF_31_31_0.txt';
filename = '~/Desktop/LCALIF_31_31_0.txt';

progress_step = 1000;

in_file = fopen(filename, 'r');

data = struct(...
   'G_E',         [],...
   'G_I',         [],...
   'G_IB',        [],...
   'dynVthRest',  [],...
   'V',           [],...
   'Vth',         [],...
   'a',           []...
);


fields = fieldnames(data);

string = 'asdf';
num = 0.0;

while(strcmp(string, '\n') == 0)
   if(mod(num, progress_step) == 0)
      disp(['Time: ', num2str(num)]);
      fflush(1);
   end
   num += 1;
   string = fgetl(in_file);
   if (feof(in_file))
      break
   end
   newStr = string;

   for i = 1:length(fields)
      tok = fields{i};
      idx = strfind(newStr, tok);
      newStr = newStr(idx+length(tok)+1:end);
      idx = strfind(newStr(2:end), ' ');
      if(isempty(idx))
         data.(fields{i}) = [data.(fields{i}); str2num(newStr(1:end))];
      else
         data.(fields{i}) = [data.(fields{i}); str2num(newStr(1:idx(1)))];
      end
   end
end

fclose(in_file);

figure;
hold on;

plot(data.V, 'b');
plot(data.Vth, 'r');
plot(data.a .* 10, 'g');
plot(data.dynVthRest, 'k');
hold off;

