function [] = pv_objects()

    global N NO NX NY DTH n_time_steps begin_step output_path input_path
    global spike_array num_fig figure_ndx

%read number of objects and number of their indices in image
num_filename = 'num.bin';
num_filename = [input_path, num_filename];
disp (num_filename);
if exist (num_filename, 'file');
    disp('ok');
    fid= fopen(num_filename,'r', 'native');
    num_fig= fread(fid, 1, 'int');

    num_indices = fread(fid, num_fig,'int');

    fclose(fid);
end

figure_ndx = cell(2,num_fig);

%read indices of objects
for i_fig = 0:num_fig-1
    numstr = int2str(i_fig);
    input_indices_file = 'figure_';
    input_indices_file = [input_indices_file,numstr];
    input_indices_file = [input_indices_file, '.bin'];
    input_indices_file = [input_path,input_indices_file];
    disp(input_indices_file);

    fid = fopen(input_indices_file, 'r', 'native');
    figure_ndx{i_fig+1}= fread(fid, num_indices(i_fig+1),'int');
    figure_ndx{i_fig+1}= figure_ndx{i_fig+1} + 1;

    fclose(fid);
end

for i_fig=1:num_fig

    figure_rate = 1000 * sum(sum(spike_array{1}(:,figure_ndx{i_fig}))) /(length(figure_ndx{i_fig}) * size(spike_array{1},1) );
    disp(['figure', num2str(i_fig),'_rate{1} = ', num2str(figure_rate)]);
    %     figure_ratei = 1000* sum(sum(spike_array{2}(:,figure_ndx{i_fig})))/(length(figure_ndx{i_fig})* size(spike_array{i_fig},1));
    %     disp(['figure',num2str(i_fig),'rate{2} =', num2str(figure_ratei)]);
end

end
