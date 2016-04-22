function [checkpoints_list] = getCheckpointList(checkpoint_parent, checkpoint_children)
      num_child_checkpoints = size(checkpoint_children,1);
      checkpoints_list = {};
      for i_child_checkpoint = 1 : num_child_checkpoints
	checkpoints_folder = ...
	    [checkpoint_parent, filesep, checkpoint_children{i_child_checkpoint,:}, filesep, "Checkpoints"];
	checkpoint_subdir_list  = glob([checkpoints_folder, filesep, "Checkpoint*"]);
	if isempty(checkpoint_subdir_list)
	  checkpoints_folder = ...
	      [checkpoint_parent, filesep, checkpoint_children{i_child_checkpoint,:}, filesep, "Last"];
	  checkpoint_subdir_list  = {checkpoints_folder};
	endif	  
	if i_child_checkpoint == 1
	  checkpoints_list = checkpoint_subdir_list;
	else
	  checkpoints_list = [checkpoints_list; checkpoint_subdir_list];
	endif
      endfor  %% i_child_checkpoint
endfunction
