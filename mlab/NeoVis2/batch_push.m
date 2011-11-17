

local_dir = pwd;

chdir("/mnt/data1/repo/neovision-results-challenge-tailwind/");

object_ids = [16]; %% [7:17,21:22,30:31];
object_name = cell(length(object_ids),1);
for i_object = 1 : length(object_name)
                 object_name{i_object} = num2str(object_ids(i_object), "%3.3i");
endfor

for i_object = 1 : length(object_name)
                 mkdir(object_name{i_object});
     chdir(object_name{i_object});
system("git pull");
system("git add .");
system(["git commit -m ", object_name{i_object}]);
system("git push");
chdir("/mnt/data1/repo/neovision-results-challenge-tailwind/");

endfor

chdir(local_dir);

