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

     still = 0;

     if still

     timerange= 29;
     timeoffset=4;
     phase=zeros(128,128);
     spikes=zeros(128,128);
     activity=zeros(timerange);

     period=10.00;


    filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/walking/%05d.png',100);
  %%  filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/input/grayimage.png');

     gray_image = imread(filename);

    
     
     spiking  = zeros(period+1); 
     spiking2d = zeros(period+1,fix(timerange./period)+1);    

     for timestep=1:timerange

     timestep     
     remainder=timestep - fix(fix(timestep./period)*period) +1 

     spike_frame = full(reshape(spike_array(timestep+timeoffset,:),[128,128]))'; %spikes

     imagesc(spike_frame);

     activity(timestep) = sum(sum(spike_frame))/256/256*1000;
     tstep(timestep)    = timestep;

     phase  = ifelse(spike_frame,phase+(period-remainder+1),phase);

     spikes = ifelse(spike_frame,spikes+1,spikes);
      
     spiking(remainder+1)=spiking(remainder+1)+sum(sum(spike_frame));
     spiking2d(remainder+1,(fix(timestep./period)+1))=sum(sum(spike_frame));
 
     end %% timestep loop     

     phases  = phase./(spikes+(spikes==0)); % avoid divide by zero

     figure(1);
     %colormap(gray);
     imagesc(gray_image);
     gray_image(1:10)
     colorbar;

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_original.png',layer,timeoffset,timeoffset+timestep);
     print(filename);

     figure(2); 
     %colormap(gray);
     plot(tstep,activity);


     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_activity.png',layer,timeoffset,timeoffset+timestep);
     print(filename);

     figure(3);
     %colormap(gray);
     imagesc(spikes);
     colorbar;

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_spikes.png',layer,timeoffset,timeoffset+timestep);
     print(filename);

     figure(4);
     %colormap(gray);
     imagesc(phase);
     colorbar;

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_phase.png',layer,timeoffset,timeoffset+timestep);
     print(filename);

     figure(5);
     %colormap(gray);
     imagesc(phases);
     colorbar;

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_phases.png',layer,timeoffset,timeoffset+timestep);
     print(filename);


     figure(6);
     %colormap(gray);
     imagesc(spiking2d);
     colorbar;

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_spiking2d.png',layer,timeoffset,timeoffset+timestep);
     print(filename);
 
     figure(7);
     plot(spiking);
     
     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/Still%02d__%03d_%03d_spiking.png',layer,timeoffset,timeoffset+timestep);
     print(filename);

keyboard;
keyboard;

endif; %% of if still =============================================================

imagestart = 1;
imagestop  = 100;
timestepwidth   = 10;
maxint = 2;

      allsteps = imagestop*timestepwidth

      activity   = zeros(1,allsteps);  % all spikes
      activityB  = zeros(1,allsteps);  % background
      activityA  = zeros(1,allsteps);  % spikes in the amoeba
    
      

      frequency   = zeros(1,allsteps);  % all spikes
      frequencyB  = zeros(1,allsteps);  % background
      frequencyA  = zeros(1,allsteps);  % spikes in the amoeba
       

      activityM  = zeros(1,allsteps);  % moving spikes
      activityL  = zeros(1,allsteps);  % Large spikes
      activityS  = zeros(1,allsteps);  % Small spikes

      activityT  = zeros(1,allsteps);  % total spikes to compare with all spikes
      integratedT = zeros(1,allsteps);

      tstep    = zeros(1,allsteps);

      total_spikes = zeros(128,128);

      tgray    = zeros(128,128,maxint);  % array of the last 20 spiking
	        		     	% frames of the 128x128 ganglion cells
 
for imagestep=imagestart:imagestop        % loop over images, each has 50 time steps

     imagestep

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/input/amoebamovie/mamoeba_1f_8_64_%03d.png',imagestep);
				%
     maskname=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/input/amoebamovie/mask_1f_8_64_%03d.png',imagestep);
%     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/input/amoeba/1f/sigma1/amoeba_1f_1_64_same_gjk.png'); 

     gray_image = imread(filename);

     mask_image = imread(maskname);

     size(mask_image);

     amoeba_mask = imresize(mask_image,[128 128]);
     amoeba_mask_size = nnz(amoeba_mask);
     background_mask = uint8(~amoeba_mask);
     background_mask_size = nnz(background_mask);

     %figure;
     %imagesc(amoeba_mask);
     %figure;
     %imagesc(background_mask);
     %keyboard;

     gray_image = repmat(gray_image,[1 1 3]);

    for timestep=(imagestep-1)*timestepwidth+1:(imagestep)*timestepwidth  % 50 steps per frame
%     for timestep=1:timestepwidth

     timestep

     color_image = imresize(gray_image,[128 128]);  % recopy

     size(spike_array);

     spike_frame = full(reshape(spike_array(timestep,:),[128,128]))'; %spikes

     spike_frame_size = numel(spike_frame);

     %%imagesc(spike_frame);
     color_image(:,:,1) = ifelse(spike_frame,255,color_image(:,:,1));

     color_image(:,:,2) = ifelse(spike_frame,0,color_image(:,:,2));

     color_image(:,:,3) = ifelse(spike_frame,0,color_image(:,:,3));

     

     %imagesc(color_image);
     title(num2str(timestep));
     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/amoeba/a_small_8_64_t_%02d_%05d_o.png',layer,timestep);
     %%print(filename);
     
     imwrite(color_image,filename);

 

     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/amoeba/a_small_8_64_t_%02d_%05d_s.png',layer,timestep);
     %%print(filename);
     imwrite(spike_frame,filename);

     total_spikes = total_spikes+spike_frame; 

%%     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/amoeba/a_small_8_64_t_%02d_%05d_t.png',layer,timestep);
     %%print(filename);
%%     imagesc(total_spikes);
%%     print(filename,'-dpng');

     %size(spike_frame)
     %size(background_mask)
     %size(amoeba_mask)

     activity(timestep)  = sum(spike_frame(:));
 
     activityB(timestep) = sum(sum(spike_frame.*background_mask));
     activityA(timestep) = sum(sum(spike_frame.*amoeba_mask));
     tstep(timestep)     = timestep;
  
     frequency(timestep) = activity(timestep)/spike_frame_size*1000;
     frequencyB(timestep) = activityB(timestep)/(background_mask_size+(background_mask_size==0))*1000;
     frequencyA(timestep) = activityA(timestep)/(amoeba_mask_size+(amoeba_mask_size==0))*1000;

     disp(["Timestep ",num2str(timestep)," All ",num2str(activity(timestep))," Background ",num2str(activityB(timestep))," Amoeba ",num2str(activityA(timestep))]);

     disp(["Timestep ",num2str(timestep)," All ",num2str(frequency(timestep))," Background ",num2str(frequencyB(timestep))," Amoeba ",num2str(frequencyA(timestep))]);

     

%     for imagecopy=1:maxint-1
%     tgray(:,:,imagecopy) = tgray(:,:,imagecopy+1);
%     endfor % of the for loop
%     tgray(:,:,maxint) = spike_frame;

%     integrated_spikes = zeros(128,128);  
%     for imagecopy=1:maxint
%     integrated_spikes = integrated_spikes + tgray(:,:,imagecopy);
%     endfor % of the for loop

%      spike_frame = integrated_spikes; % override here to use code below

%      neighbour_spikes = spike_frame;
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[-1,-1]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[-1, 0]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[-1, 1]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 0,-1]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 0, 0]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 0, 1]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 1,-1]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 1, 0]);
%      neighbour_spikes = neighbour_spikes + circshift(spike_frame,[ 1, 1]);

     %imagesc(spike_frame);
     %filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/spike%02d_%05d.png',layer,timestep);
     %print(filename);
     %imwrite(spike_frame,filename);  


%     imagesc(integrated_spikes);
%     colorbar;
%     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/IntegratedSpikeDW%02d_%05d.png',layer,timestep);
%     print(filename);


%     imagesc(neighbour_spikes);
%     colorbar;
%     filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/NeighbourSpikeDW%02d_%03d_%05d.png',layer,maxint,timestep);
%     print(filename);

%     imwrite(integrated_spikes,filename);  

%      for xx=1:128
%       for yy=1:128
%       if (yy>64)
%         if (xx<64)
%         activityL(timestep) = activityL(timestep)+spike_frame(yy,xx);
%         endif
%         if (xx>64)
%         activityS(timestep) = activityS(timestep)+spike_frame(yy,xx);
%         endif
%       endif 
%         activityT(timestep) = activityT(timestep)+spike_frame(yy,xx);
% 
%       endfor
%      endfor

  integratedT(timestep) = sum(total_spikes(:));
  
%  imagesc(total_spikes); colorbar;

%  sum(sum(total_spikes))

  end % timestep

%figure()
%imagesc(total_spikes); colorbar;
%title(num2str(timestep))
%sum(sum(total_spikes))

end % imagestep

figure()
imagesc(total_spikes); 
colorbar;
title(num2str(timestep))
sum(sum(total_spikes))
print("total_spikes.pdf","-dpdf");

figure();
plot(tstep,frequency,tstep,frequencyB+100,tstep,frequencyA+200);
ymax = (max(frequencyA(:))+200)*1.5+1;
axis([0,1000,0,ymax]);
grid;
print("frequency.pdf","-dpdf");


figure();
plot(tstep,integratedT);
ymax = max(integratedT(:))*1.5+1;
axis([0,1000,0,ymax]);
print("integrated.pdf","-dpdf");


%figure(2)
%plot(tstep,activityT);
%figure(3)
%plot(tstep,activityL);
%filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/LargeSpikeActivity%02d.png',layer)
%print(filename);
%figure(4)
%plot(tstep,activityS);
%filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/SmallSpikeActivity%02d.png',layer)
%print(filename);
%figure(5)
%plot(tstep,activityL,"0",tstep,activityS,"1");
%filename=sprintf('/Users/gerdjkunde/Documents/workspace/gjkunde/output/movie/OverlaySpikeActivity%02d.png',layer)
%print(filename);

keyboard;
    
       
    %% accumulate rate info
    rate_array{layer} = rate_array{layer} + ...
        1000 * full( mean(spike_array(stim_steps,:),1) ) / DELTA_T;
    
  endfor %% % i_epoch
  
