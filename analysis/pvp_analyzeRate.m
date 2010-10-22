function [rate_array] = ...
      pvp_analyzeRate(layer, ...
		      epoch_struct, ...
		      layer_struct, ...
		      rate_array)

  global BIN_STEP_SIZE DELTA_T

  %% init rate array
  rate_array{layer} = zeros(1, layer_struct.num_neurons(layer));

  stim_steps = ...
      epoch_struct.stim_begin_step(layer) : epoch_struct.stim_end_step(layer);

  %% start loop over epochs
  for i_epoch = 1 : epoch_struct.num_epochs
    disp(['i_epoch = ', num2str(i_epoch)]);
    
    %% read spike train for this epoch
    [spike_array] = ...
        pvp_readSparseSpikes(layer, ...
			     i_epoch, ...
			     epoch_struct, ...
			     1);
    if isempty(spike_array)
      continue;
    endif %%
    
    %% accumulate rate info
    rate_array{layer} = rate_array{layer} + ...
        1000 * full( mean(spike_array(stim_steps,:),1) ) / DELTA_T;
    
  endfor %% % i_epoch
  
