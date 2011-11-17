%%object_name = {"007";  "008";  "009";  "010";  "011"; "012";  "013";  "014"; "015";  "016";  "017";  "021";  "022";  "030";  "031"};

object_ids = [7:17,21:22,30:31]; %% [26:50];
object_name = cell(length(object_ids),1);
for i_object = 1 : (length(object_name)-1)
		 object_name{i_object} = num2str(object_ids(i_object), "%3.3i");		 
endfor
object_name{length(object_name)} = num2str(object_ids(length(object_name)), "%3.3i");		 

for i_object = 1 : 0 %%length(object_name)
padChips([], ...
	 object_name{i_object}, ...
	       [], ...
	       [], ...
	       [], ...
	       [], ...
	       [], ...
	       [], ...
	       [])
endfor

for i_object = 1 : 0 %% length(object_name)
		 chipFileOfFilenames([], ...
				     object_name{i_object}, ...
				     [], ...
				     [], ...
				     [], ...
				     [], ...
				     [], ...
				     [], ...
				     []);
endfor

for i_object = 1 : 0%% length(object_name)
		 %%mkdir(["~/workspace-indigo/Clique2/input/Tailwind/Challenge/", object_name{i_object}, filesep])
%%mkdir(["~/workspace-indigo/Clique2/input/Tailwind/Challenge/", object_name{i_object}, filesep, "Car3", filesep])
mkdir(["~/workspace-indigo/Clique2/input/Tailwind/Challenge/", object_name{i_object}, filesep, "Car3", filesep, "canny2", filesep])

endfor

		 for i_object = 1 : 0%% length(object_name)
				  %%mkdir(["/mnt/data1/repo/neovision-programs-petavision/Tailwind/Challenge/activity/", object_name{i_object}, filesep])
%%mkdir(["/mnt/data1/repo/neovision-programs-petavision/Tailwind/Challenge/activity/", object_name{i_object}, filesep, "Car3", filesep])
mkdir(["/mnt/data1/repo/neovision-programs-petavision/Tailwind/Challenge/activity/", object_name{i_object}, filesep, "Car3", filesep, "canny2", filesep])

endfor


  base_dir = ["~/workspace-indigo/Clique2/input/Tailwind/Challenge/","007", filesep, "Car3", filesep];
  base_name = ["Tailwind_", "007", "_Car3_"]; 
 for i_object = 1 : length(object_name)
  derived_dir = ["~/workspace-indigo/Clique2/input/Tailwind/Challenge/",object_name{i_object}, filesep, "Car3", filesep];
  derived_name = ["Tailwind_", object_name{i_object}, "_Car3_"]; 
  copyfile([base_dir, "canny", filesep, base_name, "canny", ".params"], [derived_dir, "canny", filesep, derived_name, "canny", ".params"])
 endfor
