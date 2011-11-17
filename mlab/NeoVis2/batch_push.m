

local_dir = pwd;

chdir("/mnt/data1/repo/neovision-results-challenge-heli/");

object_ids = [26:50]; %% [7:17,21:22,30:31]; %%                                                                                                                  
object_name = cell(length(object_ids),1);                                                                                                                       
for i_object = 1 : (length(object_name)-1)                                                                                                                      
                 object_name{i_object} = num2str(object_ids(i_object), "%3.3i");                                                                                
endfor                                                                                                                                                          
object_name{length(object_name)} = num2str(object_ids(length(object_name)), "%3.3i");                                                                           

for i_object = 1 : length(object_name)
                 mkdir(object_name{i_object});
     chdir(object_name{i_object});
system("git pull");
system("git add .");
system(["git commit -m ", object_name{i_object}]);
system("git push");
chdir("/mnt/data1/repo/neovision-results-challenge-heli/");

endfor

chdir(local_dir);
