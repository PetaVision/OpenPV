
train_dir_names = {"2FC"; "4FC"; "6FC"; "8FC"};
object_name = cell(3);
object_name{1} = "a";
object_name{2} = "d";
object_name{3} = "t";

chip_path = ["~/Pictures/amoeba/256", filesep]; 
num_train = repmat(-1, 16, 1);
num_output_files = length(num_train);
skip_train_images = num_output_files;
begin_train_images = 1;
list_dir = "list_";
shuffle_flag = 0;

for i_object = 1 : length(object_name)
  for i_train = 1 : length(train_dir_names)
    [train_filenames, ...
     tot_train_images, ...
     tot_time, ...
     rand_state] = ...
	chipFileOfFilenames2(chip_path, ...
			     object_name, ...
			     num_train, ...
			     skip_train_images, ...
			     begin_train_images, ...
			     train_dir_names{i_train}, ...
			     [list_dir, train_dir_names{i_train}], ...
			     shuffle_flag, ...
			     []);
    endfor
endfor



%% make activity dirs
for i_object = 1 : length(object_name)
  for i_train = 1 : length(train_dir_names)
    mkdir(["/mnt/data1/amoeba/3way/activity/", train_dir_names{i_train}, filesep, object_name{i_object}, filesep])
  endfor
endfor

%% make test params dirs
for i_object = 2 : length(object_name)
  for i_train = 1 : length(train_dir_names)
    mkdir(["~/workspace-indigo/Clique2/input/amoeba/3way/", train_dir_names{i_train}, filesep, object_name{i_object}, filesep, "test", filesep])
  endfor
endfor

%% make train params dirs
for i_object = 1 : length(object_name)-1
  for i_train = 1 : length(train_dir_names)
    for i_file = 1 : num_output_files
      mkdir(["~/workspace-indigo/Clique2/input/amoeba/3way/", train_dir_names{i_train}, filesep, object_name{i_object}, filesep, "train", num2str(i_file, "%2.2i"), filesep])
    endfor
  endfor
endfor
