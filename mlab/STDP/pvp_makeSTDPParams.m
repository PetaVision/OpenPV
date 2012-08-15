function [pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDPParams(DATASET_ID, ...
		      pvp_input_path, ...
		      pvp_params_template, ...
		      pvp_frame_size, ...
		      pvp_num_frames, ...		      
		      pvp_params_filename, ...
              pvp_output_path)
  
  %%keyboard;
  global PVP_VERBOSE_FLAG
  if ~exist("PVP_VERBOSE_FLAG") || isempty(PVP_VERBOSE_FLAG)
    PVP_VERBOSE_FLAG = 0;
  endif

  global pvp_home_path
  global pvp_workspace_path
  global pvp_mlab_path
  global pvp_project_path
  global params

  if isempty(pvp_home_path)
    pvp_home_path = [filesep, "Users", filesep, "rcosta", filesep];
  endif

  if isempty(pvp_workspace_path)
    pvp_workspace_path = [pvp_home_path, "Documents", filesep,  "workspace", filesep];
  endif

  if isempty(pvp_mlab_path)
    pvp_mlab_path = [pvp_home_path, "workspace", filesep, "PetaVision", filesep, "mlab", filesep];
  endif

  if isempty(pvp_project_path)
    pvp_project_path = [pvp_workspace_path, "HyPerSTDP", filesep];
  endif

  more off;
  begin_time = time();

  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("DATASET_ID") || isempty(DATASET_ID)
    DATASET_ID = "orient_simple"; %%OlshausenField_raw32x32_tiny 
  endif
  dataset_id = tolower(DATASET_ID); %% 


  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_input_path") || isempty(pvp_input_path)
    pvp_input_path = [pvp_project_path, "input"];
  endif
  
  
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_params_template") || isempty(pvp_params_template)
    pvp_params_template = [pvp_project_path, "templates", filesep, DATASET_ID, "_", "template.params"];
  else
    pvp_params_template = [pvp_project_path, "templates", filesep, pvp_params_template, "_", "template.params"];
  endif

  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_frame_size") || isempty(pvp_frame_size)
    pvp_frame_size =  32; %% default: 32x32
    disp(["frame_size = ", num2str(pvp_frame_size)]);
  endif

  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_num_frames") || isempty(pvp_num_frames)
    pvp_num_frames =  1000; %%
    disp(["num_frames = ", num2str(pvp_num_frames)]);
  endif

  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_params_filename") || isempty(pvp_params_filename)
    pvp_params_filename = [DATASET_ID, "_", num2str(pvp_frame_size) "x" num2str(pvp_frame_size), ".params"];
  endif

  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_output_path") || isempty(pvp_output_path)
    pvp_output_path = [pvp_project_path , "output", filesep, DATASET_ID];
    mkdir(pvp_output_path);
  endif
  
  pvp_params_file = [pvp_input_path, filesep, pvp_params_filename];
  pvp_params_fid = fopen(pvp_params_file, "w");
  if pvp_params_fid < 0
    disp(["fopen failed: ", pvp_params_file]);
    return;
  end
  
  pvp_template_fid = fopen(pvp_params_template, "r");
  if pvp_template_fid < 0
    disp(["fopen failed: ", pvp_params_template]);
    return;
 end

  pvp_params_token_left = "$$$_";
  pvp_params_token_right = "_$$$";
  pvp_params_hash = ...
      {"numSteps", "numSteps", num2str(pvp_num_frames); ...
       "outputPath", "outputPath", ["""", pvp_output_path, """"]; ...
       "imageListPath", "imageListPath", ["""", pvp_project_path, "input", filesep, DATASET_ID, ".txt", """"];          
          ...
       "checkpointRead", "checkpointRead", params{1}; ...
       "checkpointWrite", "checkpointWrite", params{2}; ...
       "checkpointReadDir", "checkpointReadDir", ["""", pvp_output_path, filesep, "checkpoints", """"]; ...
       "checkpointWriteDir", "checkpointWriteDir", ["""", pvp_output_path, filesep, "checkpoints", """"]; ...
       "checkpointReadDirIndex", "checkpointReadDirIndex", params{4}; ...
       "checkpointWriteStepInterval", "checkpointWriteStepInterval", params{5}; ...
       "printParamsFilename", "printParamsFilename", ["""", params{3}, ".params", """"]; ...
       "plasticityFlag", "plasticityFlag", params{6};...
       "displayPeriod", "displayPeriod", params{7};...
       "strength_Image2Retina", "strength", params{8};...

       "wMaxInitSTDP", "wMaxInit", params{9};...
       "wMinInitSTDP", "wMinInit", params{10};...
       "tauLTP", "tauLTP", params{11};...
       "tauLTD", "tauLTD", params{12};...
       "ampLTP", "ampLTP", params{13};...
       "ampLTD", "ampLTD", params{14};...
       };

  pvp_num_params = size(pvp_params_hash, 1);
       
  %%keyboard;
  while(~feof(pvp_template_fid))
    pvp_template_str = fgets(pvp_template_fid);
    pvp_params_str = pvp_template_str;
    for pvp_params_ndx = 1 : pvp_num_params
      pvp_str_ndx = ...
	  strfind(pvp_template_str, ...
		  [pvp_params_token_left, ...
		   pvp_params_hash{pvp_params_ndx, 1}, ...
		   pvp_params_token_right]);
      if ~isempty(pvp_str_ndx)
	pvp_hash_len = ...
	    length(pvp_params_hash{pvp_params_ndx, 1}) + ...
	    length(pvp_params_token_left) + ...
	    length(pvp_params_token_right);
	pvp_template_len = ...
	    length(pvp_template_str);
	pvp_prefix = pvp_template_str(1:pvp_str_ndx-1);
	pvp_suffix = pvp_template_str(pvp_str_ndx+pvp_hash_len:pvp_template_len-1);
	pvp_params_str = ...
	    [pvp_prefix, ...
	     pvp_params_hash{pvp_params_ndx, 2}, ...
	     " = ", ...
	     num2str(pvp_params_hash{pvp_params_ndx, 3}), ...
	     pvp_suffix, ";", "\n"];
	break;
      endif
    endfor  %% pvp_params_ndx
    if(PVP_VERBOSE_FLAG)
     pvp_params_str
    end
    fputs(pvp_params_fid, pvp_params_str);
    %%keyboard;
  endwhile
  fclose(pvp_params_fid);
  fclose(pvp_template_fid);

endfunction %% pvp_makeParams
