function [test_array, rest_array] = ...
     roc_layerAnalysisAllNeurons(layer, testTime, restTime, moveTime, BEGIN_STEP, END_STEP)

% Make the global parameters available at the command-line for convenience.
global NX NY n_time_steps
global input_dir output_dir conn_dir image_dir output_path input_path


% Read relevant file names and scale parameters
[f_file, v_file, w_file, w_last, l_name, xScale, yScale] = stdp_globals( layer );

plot_rate = 0;
parse_tif = 1;
print_unique_rates = 0;


NXscaled = NX * xScale;  
NYscaled = NY * yScale;
 

% makes ROC curves to measure detection performance


begin_step = BEGIN_STEP;  % where we start the analysis 
end_step   = 0;  % not used: we use begin_step, testTime, and restTime


% open weights file


filename = [input_dir, f_file];
fid = fopen(filename, 'r', 'native');
[time,numParams,NX,NY,NF] = readHeader(fid);

N = NX * NY * NF;

i_step = 0;        
% Read spikes before BEGIN_STEP
while i_step  <= BEGIN_STEP

        if (feof(fid))
            eofstat = feof(fid);
            fprintf('feof reached: n_time_steps = %d eof = %d\n',...
                i_step,eofstat);
            break;
        else
            time = fread(fid,1,'float64');
            %fprintf('time = %f\n',time);
            num_spikes = fread(fid, 1, 'int');
            %eofstat = feof(fid);
            %fprintf('eofstat = %d\n', eofstat);
        end

        S =fread(fid, num_spikes, 'int'); % S is a column vector
        
        i_step = i_step + 1;

end % read spikes before BEGIN_STEP


nxMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
nyMar = 8; % this is 0.5* nxp for pre-syn to post-syn connections
           %  (retina to L1)
           
           
test = zeros(1,20);  % records the number of spikes during test
rest = zeros(1,20);  % and rest periods

test_array = zeros(1,N);
rest_array = zeros(1,N);  

onoff_index = 0; % counts the number of on/off experiments
                 % for a fixed image
    
%% plot image at the begining of this experiment

    %% plot new image
    if begin_step == 0
        begin_step = 0;
        image_time = time / (2*1000);
        if parse_tif
            tif_path = [image_dir 'Bars_' num2str(image_time) '.tif'];
            fprintf('%s\n',tif_path);
            figure('Name',['Image Time = ' num2str(image_time)]);
            pixels = imread(tif_path); % NX x NY aray
            imagesc(pixels)
            colormap(gray)
            axis square
            axis off
            %[targ, Xtarg, Ytarg] = stdp_parseTiff( tiff_path );
            %disp('parse tiff -> done');
            %pause
        end


    end

while i_step <= BEGIN_STEP + END_STEP
    
    
        
    % update on/off experiment index 
    onoff_index = onoff_index + 1;
    
    %% read  spikes during test interval
    test_array(:) = 0;
    while i_step  <= begin_step + 2* testTime

        if (feof(fid))
            n_time_steps = i_step - 1;
            eofstat = feof(fid);
            fprintf('feof reached: n_time_steps = %d eof = %d\n',...
                n_time_steps,eofstat);
            break;
        else
            time = fread(fid,1,'float64');
            %fprintf('time = %f\n',time);
            num_spikes = fread(fid, 1, 'int');
            eofstat = feof(fid);
            %fprintf('eofstat = %d\n', eofstat);
        end

        % average rates during testTime (image on)

        S =fread(fid, num_spikes, 'int'); % S is a column vector
        test_array(S+1) = test_array(S+1) + 1;
        
        i_step = i_step + 1;

    end % read spikes during test interval

    % find unique elements in the test rate array
    if print_unique_rates
        b= unique(test_array);
        fprintf('test unique rates: ');
        for i=1:length(b)
            fprintf('%d ',b(i));
        end
        fprintf('\n');
    end

    % now plot spiking rates during test interval

    if plot_rate
        figure('Name',['Average Rate for ',l_name, ' testTime']);
        recon2D = reshape( test_array, [NXscaled, NYscaled] );
        %     recon2D = rot90(recon2D);
        %     recon2D = 1 - recon2D;
        %figure('Name','Rate Array ');
        imagesc( recon2D' );  % plots recon2D as an image
        colorbar
        axis square
        axis off
        %pause
    end

    
    
    % select highest firing non-margin neuron when begin_step = 0
    if begin_step == 0
                   
        % find neuron with largest firing rate
        % that is not a boundary neuron
        [sortR, sortI] = sort(test_array,'descend');
        
        
        for i=1:length(sortI)
            k = sortI(i);  % linear index
            I = mod(k-1,NXscaled);
            J = (k-1-I) / NXscaled;
            % check if not boundary neuron
                if J >= nyMar && J <= (NYscaled-nyMar-1) ...
                        && I >= nxMar && I <= (NXscaled-nxMar-1)
                    maxRate = test_array(sortI(i));
                    maxI = sortI(i);
                    fprintf('max rate (k= %d I= %d, J = %d)\n',k-1,I,J);
                    break
                end
        end
               
        
        for i=1:length(w_file)
            [PATCH, patch_size, NXP, NYP] = ...
                roc_readPatch(w_file{i}, I, J, NXscaled, NYscaled);
            figure('Name',[l_name,' patch ',num2str(i)]);
            PATCH = reshape(PATCH,[NXP,NYP]);
            imagesc(PATCH,'CDataMapping','direct');
            colorbar
            axis square
            axis off
        end
        pause
        
    end
    
    
    begin_step = begin_step + 2*testTime;

    %% average rates during restTime (image off)
        
    rest_array(:) = 0;
    while i_step  <= begin_step + 2* restTime

        if (feof(fid))
            n_time_steps = i_step - 1;
            eofstat = feof(fid);
            fprintf('feof reached: n_time_steps = %d eof = %d\n',...
                n_time_steps,eofstat);
            break;
        else
            time = fread(fid,1,'float64');
            %fprintf('time = %f\n',time);
            num_spikes = fread(fid, 1, 'int');
            eofstat = feof(fid);
            %fprintf('eofstat = %d\n', eofstat);
        end

        % average rates during testTime (image on)


        S =fread(fid, num_spikes, 'int'); % S is a column vector
        rest_array(S+1) = rest_array(S+1) + 1;

        i_step = i_step + 1;

    end % read spikes during rest interval
    
    %% find unique elements in the rate array
    if print_unique_rates
        b= unique(rest_array);
        fprintf('rest unique rates: ');
        for i=1:length(b)
            fprintf('%d ',b(i));
        end
        fprintf('\n');
    end

    %% now plot rates

    if plot_rate
        figure('Name',['Average Rate for ',l_name,' restTime']);
        recon2D = reshape( rest_array, [NXscaled, NYscaled] );
        %     recon2D = rot90(recon2D);
        %     recon2D = 1 - recon2D;
        %figure('Name','Rate Array ');
        imagesc( recon2D' );  % plots recon2D as an image
        colorbar
        axis square
        axis off
        %pause
    end

    fprintf('%d: time = %f test = %f rest = %f\n',...
        onoff_index,time, test_array(maxI), rest_array(maxI));
    
    % update test and rest arrays
    num_spikes = test_array(maxI);
    test(num_spikes+1) = test(num_spikes+1)+1;
    num_spikes = rest_array(maxI);
    rest(num_spikes+1) = rest(num_spikes+1)+1;
    
    
    begin_step = begin_step + 2*restTime;
    end_step = end_step + testTime + restTime;
    
    %pause
    if plot_rate
      close all
    end
    
    
    %% check for moveTime when image changes / switches
    % Compute ROC curve
    if mod(begin_step, 2*moveTime ) == 0
        fprintf('rate distribution time = %f s\n',time/1000);
        % plot distribution of rest and test firing spikes
        % for the chosen neuron
        figure('Name','Test/Rest Firing Rates');
        subplot(1,3,1);
        bar(0:(length(test)-1), test, 'r')
        title('50ms test interval');
        subplot(1,3,2)
        bar(0:(length(rest)-1), rest, 'b')
        title('50ms rest interval');
        % build ROC curve
        subplot(1,3,3)
        N = sum(test)+sum(rest);
        NP = sum(test);
        NN = sum(rest);
        fprintf('%d positive samples\n', NP);
        fprintf('%d negative samples\n', NN);
        c=colormap(jet);
        % k=1 is no spikes (no signal)
        for k=2:length(test) % loop over threshold
             % classifying positive instances correctly among all positive 
             % samples available during the test. 
            tp = sum(test(k:end)) / NP; % true positive
            % how many incorrect positive results occur among all negative 
            % samples available during the test.
            fp = sum(rest(k:end)) / NN; % false positive
            plot([fp],[tp],'o','MarkerFaceColor',c(3*k,:),...
                'MarkerEdgeColor',c(3*k,:));
            if k==2
                hold on
                axis([0 1 0 1]);
                xlabel('False Positive Rate');
                ylabel('True Positive Rate');
                title('ROC curve');
            end
        end
        pause
    end
    
end % while loop over i_step


%
% end primary function
    
    
function [time,numParams,NX,NY,NF] = readHeader(fid)
% NOTE: see analysis/python/PVReadWeights.py for reading params
% We call this function first because it rewinds the input file

    head = fread(fid,3,'int');
    if head(3) ~= 2  % PV_INT_TYPE
       disp('incorrect file type')
       return
    end
    numParams = head(2);
    fseek(fid,0,'bof'); % rewind file
    params = fread(fid, numParams-2, 'int'); 
    %pause
    NX         = params(4);
    NY         = params(5);
    NF         = params(6);
    %fprintf('numParams = %d ',numParams);
    %fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
    %pause
    % read time - last two params
    time = fread(fid,1,'float64');
    %fprintf('time = %f\n',time);
    %pause
    
% End subfunction 
%    

