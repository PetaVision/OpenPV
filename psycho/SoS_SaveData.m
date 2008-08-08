SoS_data.SoS_duration = SoS_duration(1:SoS_trial);
SoS_data.SoS_target_label = SoS_target_label(1:SoS_trial);
SoS_data.SoS_control_label = SoS_control_label(1:SoS_trial);
SoS_data.SoS_choice = SoS_choice(1:SoS_trial);
SoS_data.SoS_target_flag = SoS_target_flag(1:SoS_trial);
SoS_data.VBLTimestamp = VBLTimestamp(1:SoS_trial,:);
SoS_data.StimulusOnsetTime = StimulusOnsetTime(1:SoS_trial,:);
SoS_data.FlipTimestamp = FlipTimestamp(1:SoS_trial,:);
SoS_data.Missed = Missed(1:SoS_trial,:);
SoS_data.Beampos = Beampos(1:SoS_trial,:);
if SoS_image_source == SoS_IMAGE_FROM_DATABASE
    [SoS_data.SoS_DB_file{1:SoS_trial}] = deal(SoS_DB_file);
elseif SoS_image_source == SoS_IMAGE_FROM_RENDER
    SoS_data.SoS_radius_x = SoS_radius_x(1:SoS_trial);
    SoS_data.SoS_radius_y = SoS_radius_y(1:SoS_trial);
    SoS_data.SoS_offset_x = SoS_offset_x(1:SoS_trial);
    SoS_data.SoS_offset_y = SoS_offset_y(1:SoS_trial);
end
cd (SoS_data_path);
save(SoS_file_new, 'SoS_data');
cd (SoS_src_path);