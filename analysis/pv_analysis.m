% this reads directly the neuron acitvity.

% Make the global parameters available at the command-line for convenience.
global N NK NO NX NY n_time_steps begin_step
global spike_array num_target rate_array target_ndx vmem_array
global input_dir output_path input_path

bin_output = 0;
tif_output = 1;

bin_input = 1;
tif_input = 0;

pv_dir = '/nh/home/manghel/petavision/workspace/pv-craig/';
input_dir = [pv_dir 'src/io/input/'];
%out_dir = [pv_dir 'src/output/vstripes1/'];
%out_dir = [pv_dir 'src/output/vstripes4/'];
%out_dir = [pv_dir 'src/output/holed_rectangle1/'];
out_dir = [pv_dir 'src/output/'];
    
image_file = [input_dir 'vstripes1_64x64.bin'];
%image_file = [input_dir 'vstripes4_64x64.bin'];
%image_file = [input_dir 'holed_rectangle_64x64.bin'];
    
NX = 64;
NY = 64;
N = NX * NY;
NO = 8;    % number of orientations
NO_retina = 1;
NK = 1;    % number of curvatures
n_time_steps = 1;

plot_image = 1;
plot_input = 0;  % from Garr's code
plot_retina = 1;
plot_output = 1;

f=0;        % figure index

% plot input image

if plot_image
    if bin_input
        fprintf('read and plot bin image\n');
        fid = fopen(image_file, 'r');
        input_image = fread(fid,[NX,NY],'float32');
        %input_image = input_image * 255;
        fclose(fid);
        f=f+1;
        figure(f);
        imagesc(input_image);% plots input_image as an image
                         % Each element of input_image corresponds 
                         % to a rectangular area in the image
        colormap(gray);
        pause
    end
    
    if tif_input
        fprintf('read and plot tiff image\n');
        input_image = imread(image_file,'tif');
        f=f+1;
        figure(f);
        imagesc(input_image);% plots input_image as an image
        % Each element of input_image corresponds
        % to a rectangular area in the image
        colormap(gray);
        pause
    end

end


if plot_input % using Garr's code
    
    fid = fopen(image_file, 'r');
    input_image = fread(fid,[NX,NY],'float32');
    fclose(fid);
    %size(input_image)
    %pause
    input_image = input_image * 255;
    %input_image = sum(input_image,3)';
    %size(input_image)
    %pause
    plot_title = 'input image';
    [NX, NY] = size(input_image);
    NK = 1;
    NO = 1;
    N = NX * NY;
    pv_reconstruct( input_image(:), plot_title ); 
    % input_image(:) transforms the MxN matrix into a MNx1 vector,
    % column 1, followed by column 2, followed by column 3, etc.
    pause
end


% plot output of retina (layer 0)

if plot_retina
    
    if bin_output
        file_name = [out_dir 'f0.bin'];
        fid = fopen(file_name, 'r');
    else
        file_name = [out_dir 'f0.tif'];
        k=0;
    end
    
    fprintf('plot retina from %s:\n', file_name);
    f=f+1;
    figure(f);
    f_old =f;
    
    for t=1:n_time_steps
        
        f=f_old-1;
        fprintf('time %d:\n',t);
        
        for i=1:NO_retina
            if bin_output
                pixels = fread(fid,[NX,NY],'float32');
            else
                k=k+1;
                pixels = imread(file_name,k);
            end
            
            f=f+1;
            figure(f);
                
            if ~isempty(pixels)
                imagesc(pixels);% plots input_image as an image
                colormap(gray);
            else
                fprintf('empty pixels: end of output file\n');
                break;
            end
            
            fprintf('%d %f\n',i,sum(sum(pixels)) );
            
        end % feature loop
        pause
    end % time loop
    
end


% plot output of various layers

if plot_output
    
    if bin_output
        file_name = [out_dir 'f1.bin'];
        fid = fopen(file_name, 'r');
    else
        file_name = [out_dir 'f1.tif'];
        k=0;
    end
    
    fprintf('plot output from %s:\n', file_name);
    f=f+1;
    figure(f);
    f_old =f;
   
    
    for t=1:n_time_steps
        
        f=f_old;
        fprintf('time %d:\n',t);
        
        for i=1:NO
            
            if bin_output
                pixels = fread(fid,[NX,NY],'float32');
            else
                k=k+1;
                pixels = imread(file_name,k);
            end
            
            %f=f+1;
            figure(f);
            subplot(2,4,i);
            
            if ~isempty(pixels)
                imagesc(pixels);% plots input_image as an image
                colormap(gray);
            else
                fprintf('empty pixels: end of output file\n');
                break;
            end
            
            fprintf('%d %f\n',i,sum(sum(pixels)) );
            
        end  % features loop
        pause
    end % time loop
    
end
