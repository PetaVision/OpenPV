% DESCRIPTION:
%       subscript to check input structures and optional input parameters
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 21st December 2010
%       last update - 28th February 2012
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% pre-calculate the data type needed to store the matrix indices given the
% total number of grid points: indexing variables will be created using
% this data type to save memory
if kgrid.total_grid_points < intmax('uint8');
    index_data_type = 'uint8';
elseif kgrid.total_grid_points < intmax('uint16');
    index_data_type = 'uint16';
elseif kgrid.total_grid_points < intmax('uint32');
    index_data_type = 'uint32';                
else
    index_data_type = 'double';
end   

% =========================================================================
% CHECK MEDIUM STRUCTURE INPUTS
% =========================================================================

% check the medium input is defined as a structure
if ~isstruct(medium)
    error('medium must be defined as a MATLAB structure');
end

% check medium fields
checkFieldNames(medium, {'sound_speed', 'sound_speed_ref', 'density', 'alpha_coeff', 'alpha_power', 'alpha_mode', 'alpha_filter', 'alpha_sign', 'BonA'});
enforceFields(medium, {'sound_speed'});

% allow the density field to be blank if the medium is homogeneous
if ~isfield(medium, 'density') && numel(medium.sound_speed) == 1
    user_medium_density_input = false;
    medium.density = 1;
else
    enforceFields(medium, {'density'});
    user_medium_density_input = true;
end

% check medium absorption inputs
if isfield(medium, 'alpha_coeff') || isfield(medium, 'alpha_power')
    
    % if one absorption parameter is given, enforce the other
    enforceFields(medium, {'alpha_coeff', 'alpha_power'});
    
    % check y is within 0 to 3
    if medium.alpha_power > 3 || medium.alpha_power < 0
        error('medium.alpha_power must be within 0 and 3');
    end

    % display warning if y is close to 1 and the dispersion term has
    % not been set to zero
    if ~(isfield(medium, 'alpha_mode') && strcmp(medium.alpha_mode, 'no_dispersion'))
        if medium.alpha_power == 1
            error('The power law dispersion term in the equation of state is not valid for medium.alpha_power = 1. This error can be avoided by choosing a power law exponent close to, but not exactly, 1. If modelling acoustic absorption for medium.alpha_power = 1 is important and modelling dispersion is not critical, this error can also be avoided by setting medium.alpha_mode to ''no_dispersion''.');
        end
    end
    equation_of_state = 'absorbing';
    
    % check the absorption mode input is valid
    if isfield(medium, 'alpha_mode')
        if ~ischar(medium.alpha_mode) || (~strcmp(medium.alpha_mode, 'no_absorption') && ~strcmp(medium.alpha_mode, 'no_dispersion'))
            error('medium.alpha_mode must be set to ''no_absorption'' or ''no_dispersion''');
        end
    end    
    
    % check the absorption filter input is valid
    if isfield(medium, 'alpha_filter') && ~all(size(medium.alpha_filter) == size(kgrid.k))
        error('medium.alpha_filter must be the same size as the computational grid');
    end
    
    % check the absorption sign input is valid
    if isfield(medium, 'alpha_sign') && (~isnumeric(medium.alpha_sign) || numel(medium.alpha_sign) > 2)
        error('medium.alpha_sign must be given as a 2 element numerical array')
    end    
else
    equation_of_state = 'lossless';
end

% check if BonA is given and then set the nonlinear flag
if isfield(medium, 'BonA')
    nonlinear = true;
else
    nonlinear = false;
end 

% select the reference sound speed used in the k-space operator based on
% the heterogeneous sound speed map
if isfield(medium, 'sound_speed_ref')
    if isnumeric(medium.sound_speed_ref)
        c_ref = medium.sound_speed_ref;
    elseif strcmp(medium.sound_speed_ref, 'min')
        c_ref = min(medium.sound_speed(:));
    elseif strcmp(medium.sound_speed_ref, 'mean')
        c_ref = mean(medium.sound_speed(:));
    else strcmp(medium.sound_speed_ref, 'max')
        c_ref = max(medium.sound_speed(:));        
    end
else
    c_ref = max(medium.sound_speed(:));
end
disp(['  reference sound speed: ' num2str(c_ref) 'm/s']);

% =========================================================================
% CHECK SENSOR STRUCTURE INPUTS
% =========================================================================

% set default sensor flags
use_sensor = false;
time_rev = false;
compute_directivity = false;
store_time_series = true;
reorder_data = false;
binary_sensor_mask = true;
transducer_sensor = false;

% check sensor fields
if ~isempty(sensor)

    % check the sensor input is valid
    if ~(isstruct(sensor) || strcmp(class(sensor), 'kWaveTransducer'))
        error('sensor must be defined as a MATLAB structure or an object of the kWaveTransducer class');
    end
    
    % set sensor flag to true and additional default flags
    use_sensor = true;
    
    % check if sensor is a transducer, otherwise check input fields
    if ~strcmp(class(sensor), 'kWaveTransducer')
        if kgrid.dim == 2

            % check field names including the directivity inputs
            checkFieldNames(sensor, {'mask', 'directivity_pattern', 'directivity_angle', 'directivity_size',...
                'time_reversal_boundary_data', 'frequency_response', 'record_mode'});

            % check for sensor directivity input and set flag
            if isfield(sensor, 'directivity_angle')
                compute_directivity = true;
            end

        else
            % check field names without directivity inputs (these are not supported in 1 or 3D)
            checkFieldNames(sensor, {'mask', 'time_reversal_boundary_data', 'frequency_response', 'record_mode'});
        end
        
        % enfore the sensor.mask field
        enforceFields(sensor, {'mask'});

        % check for time reversal inputs and set flag
        if isfield(sensor, 'time_reversal_boundary_data')
            time_rev = true;
        end
        
        % check for the 'record_mode' input and set flags
        if isfield(sensor, 'record_mode')
            % check the input is valid
            if ~(strcmp(sensor.record_mode, 'time_history') || strcmp(sensor.record_mode, 'statistics'))
                error('sensor.record_mode must be set to ''time_history'' or ''statistics''');
            end
            
            % set the usage flags
            if strcmp(sensor.record_mode, 'statistics')
               store_time_series = false;
            end
        end
        
        % check if sensor mask is a binary grid or a set of interpolation points
        if (kgrid.dim == 3 && numDim(sensor.mask) == 3) || (kgrid.dim ~= 3 && all(size(sensor.mask) == size(kgrid.k)))

            % check the grid is binary
            if sum(sensor.mask(:)) ~= numel(sensor.mask) - sum(sensor.mask(:) == 0)
                error('sensor.mask must be a binary grid (numeric values must be 0 or 1)');
            end

        else
            
            % set Cartesian mask flag (this is modified in
            % createStorageVariables if the interpolation setting is set to
            % nearest)
            binary_sensor_mask = false;   
            
            % extract Cartesian data from sensor mask
            switch kgrid.dim
                case 1
                    sensor_x = sensor.mask;
                case 2
                    sensor_x = sensor.mask(1, :);
                    sensor_y = sensor.mask(2, :);
                case 3
                    sensor_x = sensor.mask(1, :);
                    sensor_y = sensor.mask(2, :);
                    sensor_z = sensor.mask(3, :);    
            end
            
            % compute an equivalent sensor mask using nearest neighbour
            % interpolation, if time_rev = false and cartesian_interp = 'linear'
            % then this is only used for display, if time_rev = true or
            % cartesian_interp = 'nearest' this grid is used as the sensor.mask 
            [sensor.mask, order_index, reorder_index] = cart2grid(kgrid, sensor.mask);
            
            % if in time reversal mode, reorder the p0 input data in the order
            % of the binary sensor_mask  
            if time_rev

                % append the reordering data
                new_col_pos = length(sensor.time_reversal_boundary_data(1, :)) + 1;
                sensor.time_reversal_boundary_data(:, new_col_pos) = order_index;

                % reorder p0 based on the order_index
                sensor.time_reversal_boundary_data = sortrows(sensor.time_reversal_boundary_data, new_col_pos);

                % remove the reordering data
                sensor.time_reversal_boundary_data = sensor.time_reversal_boundary_data(:, 1:new_col_pos - 1);

            end
        end
        
    else

        % if the sensor is a transducer, check that the simulation is in 3D
        if kgrid.dim ~= 3
            error('Transducer inputs are only compatible with 3D simulations');
        end
        
        % check that the transducer is only being used in forward mode
        if isfield(sensor, 'time_reversal_boundary_data')
            error('Transducer inputs not yet supported with time reversal');
        end
        
        % set transducer sensor flag
        transducer_sensor = true;
        
        % check to see if there is an elevation focus
        if isinf(sensor.elevation_focus_distance)
            transducer_receive_elevation_focus = false;
        else
            transducer_receive_elevation_focus = true;
            
            % get the elevation mask that is used to extract the correct values
            % from the sensor data buffer for averaging
            transducer_receive_mask = sensor.elevation_beamforming_mask;
        end
        
    end
end

% check for directivity inputs with time reversal
if kgrid.dim == 2 && use_sensor && compute_directivity && time_rev
    disp('WARNING: sensor directivity fields are not used for time reversal');
end

% =========================================================================
% CHECK SOURCE STRUCTURE INPUTS
% =========================================================================

% predefine default source flags
p_source = false;
ux_source = false;
uy_source = false;
uz_source = false; 
transducer_source = false;

% check source inputs
if ~(isstruct(source) ||strcmp(class(source), 'kWaveTransducer'))
    % allow an invalid or empty source input if computing time reversal,
    % otherwise return error
    if ~time_rev
        error('source must be defined as a MATLAB structure or an object of the kWaveTransducer class');
    end
elseif ~strcmp(class(source), 'kWaveTransducer')
    
    % --------------------------
    % SOURCE IS NOT A TRANSDUCER
    % --------------------------
    
    % enfore source fields
    switch kgrid.dim
        case 1
            checkFieldNames(source, {'p0', 'p', 'p_mask', 'p_mode', 'ux', 'u_mask', 'u_mode'});
        case 2
            checkFieldNames(source, {'p0', 'p', 'p_mask', 'p_mode', 'ux', 'uy', 'u_mask', 'u_mode'});
        case 3
            checkFieldNames(source, {'p0', 'p', 'p_mask', 'p_mode', 'ux', 'uy', 'uz', 'u_mask', 'u_mode'});
    end
    
    % check source input
    if isfield(source, 'p0')
        if isempty(source.p0) || ~sum(source.p0(:) ~= 0)
            % if the initial pressure is empty, remove field
            source = rmfield(source, 'p0');
        elseif ~all(size(source.p0) == size(kgrid.k))
            % throw an error if p0 is not the correct size
            error('source.p0 must be the same size as the computational grid');
        end
    end

    % check for a time varying pressure source input
    if isfield(source, 'p')

        % force p_mask to be given if p is given
        enforceFields(source, {'p_mask'});
        
        % don't allow both source.p0 and source.p in the same simulation
        % USERS: please contact us via http://www.k-wave.org/forum if this is a problem
        if isfield(source, 'p0')
            error('source.p0 and source.p can''t be defined in the same simulation');
        end
        
        % set source flag to the length of the source, this allows source.p
        % to be shorter than kgrid.t_array
        p_source = length(source.p(1, :));
        if p_source > length(kgrid.t_array)
           disp('  WARNING: source.p has more time points than kgrid.t_array, remaining time points will not be used');
        end        

        % if more than one time series is given, check the number of time
        % series given matches the number of source elements
        if (length(source.p(:,1)) > 1) && (length(source.p(:,1)) ~= sum(source.p_mask(:)))
            error('The number of time series in source.p must match the number of source elements in source.p_mask');
        end

        % create an indexing variable corresponding to the location of all
        % the source elements 
        ps_index = find(source.p_mask ~= 0);
        
        % convert the data type depending on the number of indices
        eval(['ps_index = ' index_data_type '(ps_index);']);
        
        % check the source mode input is valid
        if isfield(source, 'p_mode')
            if ~ischar(source.p_mode) || (~strcmp(source.p_mode, 'additive') && ~strcmp(source.p_mode, 'dirichlet'))
                error('source.p_mode must be set to ''additive'' or ''dirichlet''');
            end
        else
            source.p_mode = SOURCE_P_MODE_DEF;
        end        
            
    end

    % check for time varying velocity source input and set source flag
    if isfield(source, 'ux') || isfield(source, 'uy') || isfield(source, 'uz') || isfield(source, 'u_mask') 

        % force u_mask to be given
        enforceFields(source, {'u_mask'});
        
        % set source flags to the length of the sources, this allows the
        % inputs to be defined independently and be of any length
        if isfield(source, 'ux')
            ux_source = length(source.ux(1, :));
            if ux_source > length(kgrid.t_array)
               disp('  WARNING: source.ux has more time points than kgrid.t_array, remaining time points will not be used');
            end
        end
        if isfield(source, 'uy')
            uy_source = length(source.uy(1, :));
            if uy_source > length(kgrid.t_array)
                disp('  WARNING: source.uy has more time points than kgrid.t_array, remaining time points will not be used');
            end
        end
        if isfield(source, 'uz')
            uz_source = length(source.uz(1, :)) ;
            if uz_source > length(kgrid.t_array)
                disp('  WARNING: source.uz has more time points than kgrid.t_array, remaining time points will not be used');
            end          
        end    

        % if more than one time series is given, check the number of time
        % series given matches the number of source elements
        if (ux_source && (length(source.ux(:,1)) > 1)) || (uy_source && (length(source.uy(:,1)) > 1)) || (uz_source && (length(source.uz(:,1)) > 1))
            if (ux_source && (length(source.ux(:,1)) ~= sum(source.u_mask(:)))) || (uy_source && (length(source.uy(:,1)) ~= sum(source.u_mask(:)))) || (uz_source && (length(source.uz(:,1)) ~= sum(source.u_mask(:))))
                error('The number of time series in source.ux (etc) must match the number of source elements in source.u_mask');
            end
        end

        % create an indexing variable corresponding to the location of all
        % the source elements 
        us_index = find(source.u_mask ~= 0);      
        
        % convert the data type depending on the number of indices
        eval(['us_index = ' index_data_type '(us_index);']);   
        
        % check the source mode input is valid
        if isfield(source, 'u_mode')
            if ~ischar(source.u_mode) || (~strcmp(source.u_mode, 'additive') && ~strcmp(source.u_mode, 'dirichlet'))
                error('source.u_mode must be set to ''additive'' or ''dirichlet''');
            end
        else
            source.u_mode = SOURCE_U_MODE_DEF;
        end

    end
else
    % ----------------------
    % SOURCE IS A TRANSDUCER
    % ----------------------
    
    % if the sensor is a transducer, check that the simulation is in 3D
    if kgrid.dim ~= 3
        error('Transducer inputs are only compatible with 3D simulations');
    end    
    
    % get the input signal - this is appended with zeros if required to
    % account for the beamforming delays (this will throw an error if the
    % input signal is not defined)
    transducer_input_signal = source.input_signal;
        
    % get the delay mask that accounts for the beamforming delays and
    % elevation focussing; this is used so that a single time series can be
    % applied to the complete transducer mask with different delays
    delay_mask = source.delay_mask;

    % set source flag - this should be the length of signal minus the
    % maximum delay
    transducer_source = length(transducer_input_signal) - max(delay_mask(:));    
    
    % get the active elements mask
    active_elements_mask = source.active_elements_mask;
    
    % get the apodization mask if not set to 'Rectangular' and convert to a
    % linear array
    if ischar(source.transmit_apodization) && strcmp(source.transmit_apodization, 'Rectangular')
        transducer_transmit_apodization = 1;
    else
        transducer_transmit_apodization = source.transmit_apodization_mask;       
        transducer_transmit_apodization = transducer_transmit_apodization(active_elements_mask ~= 0);
    end    
        
    % create indexing variable corresponding to the active elements
    us_index = find(active_elements_mask ~= 0);
    
    % convert the data type depending on the number of indices
    eval(['us_index = ' index_data_type '(us_index);']);     
    
    % convert the delay mask to an indexing variable (this doesn't need to
    % be modified if the grid is expanded) which tells each point in the
    % source mask which point in the input_signal should be used
    delay_mask = delay_mask(active_elements_mask ~= 0);
    
    % convert the data type depending on the maximum value of the delay
    % mask and the length of the source
    max_delay = max(delay_mask(:)) + length(transducer_input_signal) + 1;
    if max_delay < intmax('uint8');
        delay_mask = uint8(delay_mask);
    elseif max_delay < intmax('uint16');
        delay_mask = uint16(delay_mask);
    elseif max_delay < intmax('uint32');
        delay_mask = uint32(delay_mask);               
    end      
    
    % move forward by 1 as a delay of 0 corresponds to the first point in
    % the input signal
    delay_mask = delay_mask + 1;
            
    % clean up unused variables
    clear active_elements_mask;
end

% =========================================================================
% CHECK KGRID STRUCTURE INPUTS
% =========================================================================

% check kgrid for t_array existance and stability
if strcmp(kgrid.t_array, 'auto')
    if ~time_rev
        % create the time array
        [kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed);
    else
        % throw error requesting for t_array
        error('kgrid.t_array must be given explicitly in time reversal mode');
    end
else
    % assign dt
    dt = kgrid.t_array(2) - kgrid.t_array(1);
        
    % check the time steps are increasing
    if dt <= 0
        error('kgrid.t_array must be monotonically increasing');
    end
    
    % check the time array is evenly spaced
    if (kgrid.t_array(2:end) - kgrid.t_array(1:end-1)) ~= dt
        error('kgrid.t_array must be evenly spaced');
    end
    
    % check kgrid.t_array for stability given medium properties
    if (numel(medium.sound_speed) > 1 || numel(medium.density) > 1) &&...
            (dt > DT_WARNING_CFL*max([kgrid.dz, kgrid.dx, kgrid.dy])/max(medium.sound_speed(:)))
        disp('  WARNING: time step may be too large for a stable simulation');
    end    
end

% assign dt to kgrid if given as a structure
if isstruct(kgrid)
    kgrid.dt = dt;
end

% shorten commonly used field names (these act only as pointers provided
% that the values aren't modified)
t_array = kgrid.t_array;
c = medium.sound_speed;
rho0 = medium.density;

% =========================================================================
% CHECK OPTIONAL INPUTS
% =========================================================================

% assign the default input parameters
cartesian_interp = CARTESIAN_INTERP_DEF;
create_log = CREATE_LOG_DEF;
data_cast = DATA_CAST_DEF;
display_mask = DISPLAY_MASK_DEF;
log_scale_comp_factor = LOG_SCALE_COMPRESSION_FACTOR_DEF;
movie_args = MOVIE_ARGS_DEF;
movie_name = MOVIE_NAME_DEF;
plot_freq = PLOT_FREQ_DEF;
plot_layout = PLOT_LAYOUT_DEF;
plot_scale = PLOT_SCALE_DEF;
plot_scale_log = LOG_SCALE_DEF;
plot_sim = PLOT_SIM_DEF;
plot_PML = PLOT_PML_DEF;
PML_inside = PML_INSIDE_DEF;
record_movie = RECORD_MOVIE_DEF;
return_velocity = RETURN_VELOCITY_DEF;
smooth_c = SMOOTH_C0_DEF;
smooth_p0 = SMOOTH_P0_DEF;
smooth_rho0 = SMOOTH_RHO0_DEF;
use_kspace = USE_KSPACE_DEF;
use_sg = USE_SG_DEF;

% assign the default input parameters that vary for different dimensions
switch kgrid.dim
    case 1
        PML_x_alpha = PML_ALPHA_DEF;
        PML_x_size = PML_SIZE_DEF;       
        stream_to_disk = false;
    case 2
        PML_x_alpha = PML_ALPHA_DEF;
        PML_y_alpha = PML_ALPHA_DEF;
        PML_x_size = PML_SIZE_DEF;
        PML_y_size = PML_SIZE_DEF;
        mesh_plot = MESH_PLOT_DEF;
        movie_type = MOVIE_TYPE_DEF;
        stream_to_disk = false;
    case 3
        PML_x_alpha = PML_ALPHA_DEF;
        PML_y_alpha = PML_ALPHA_DEF;
        PML_z_alpha = PML_ALPHA_DEF;
        PML_x_size = PML_SIZE_DEF;
        PML_y_size = PML_SIZE_DEF;
        PML_z_size = PML_SIZE_DEF;
        stream_to_disk = STREAM_TO_DISK_DEF;
        save_to_disk = SAVE_TO_DISK_DEF;
        save_to_disk_exit = SAVE_TO_DISK_EXIT_DEF;
end

% replace defaults with user defined values if provided and check inputs    
if nargin < NUM_REQ_INPUT_VARIABLES
    error('Not enough input parameters');
elseif rem(nargin - NUM_REQ_INPUT_VARIABLES, 2)
    error('Optional input parameters must be given as param, value pairs');
elseif ~isempty(varargin)
    for input_index = 1:2:length(varargin)
        switch varargin{input_index}           
            case 'CartInterp'
                cartesian_interp = varargin{input_index + 1}; 
                if ~(strcmp(cartesian_interp, 'linear') || strcmp(cartesian_interp, 'nearest'))
                    error('Optional input ''CartInterp'' must be set to ''linear'' or ''nearest''');
                end     
            case 'CreateLog'
                create_log = varargin{input_index + 1}; 
                if ~islogical(create_log)
                    error('Optional input ''CreateLog'' must be Boolean');
                end
            case 'DataCast'
                data_cast = varargin{input_index + 1};
                
                % check list of valid inputs
                if ~ischar(data_cast)
                    error('Optional input ''DataCast'' must be a string');
                elseif ~(strcmp(data_cast, 'off') || strcmp(data_cast, 'double') ...
                        || strcmp(data_cast, 'single') || strcmp(data_cast, 'gsingle') ...
                        || strcmp(data_cast, 'GPUsingle') || strcmp(data_cast, 'gdouble') ...
                        || strcmp(data_cast, 'GPUdouble'))
                    error('Invalid input for ''DataCast''');
                end
                
                % replace double with off
                if strcmp(data_cast, 'double')
                    data_cast = 'off';
                end
                
                % enforce GPUmat compatability by using wrapper for
                % GPUsingle and GPUdouble 
                if strcmp(data_cast, 'GPUsingle');
                    data_cast = 'kWaveGPUsingle';
                elseif strcmp(data_cast, 'GPUdouble');
                    data_cast = 'kWaveGPUdouble';
                end
            case 'DisplayMask'
                display_mask = varargin{input_index + 1};
                if ~(strcmp(display_mask, 'off') || all(size(display_mask) == size(kgrid.k)))
                    error('Optional input ''DisplayMask'' must be the same size as the computational grid or set to ''off''');
                end
                % force mask to be boolean
                if ~strcmp(display_mask, 'off') 
                    display_mask = (display_mask == 1);
                end
            case 'LogScale'
                plot_scale_log = varargin{input_index + 1};
                if numel(plot_freq) == 1 && isnumeric(plot_scale_log) && plot_scale_log > 0
                    log_scale_comp_factor = plot_scale_log;
                    plot_scale_log = true;
                elseif ~islogical(plot_scale_log)
                    error('Optional input ''LogScale'' must be Boolean or a single numerical value > 0');
                end              
            case 'MeshPlot'
                if kgrid.dim == 2
                    mesh_plot = varargin{input_index + 1};
                    if ~islogical(mesh_plot)
                        error('Optional input ''MeshPlot'' must be Boolean');
                    end
                else
                    error('Optional input ''MeshPlot'' only supported in 2D');
                end
            case 'MovieArgs'
                movie_args = varargin{input_index + 1};  
            case 'MovieName'
                movie_name = varargin{input_index + 1};
                if ~ischar(movie_name)
                    error('Optional input ''MovieName'' must be a string');
                end   
            case 'MovieType'
                if kgrid.dim == 2
                    movie_type = varargin{input_index + 1};
                    if ~(strcmp(movie_type, 'frame') || strcmp(movie_type, 'image'))
                        error('Optional input ''MovieType'' must be set to ''frame'' or ''image''');
                    end
                else
                    error('Optional input ''MovieType'' only supported in 2D');
                end
            case 'PlotFreq'
                plot_freq = varargin{input_index + 1}; 
                if ~(numel(plot_freq) == 1 && isnumeric(plot_freq) && (round(plot_freq) == plot_freq) && (plot_freq > 0))
                    error('Optional input ''PlotFreq'' must be a single positive scalar value');
                end
            case 'PlotLayout'
                plot_layout = varargin{input_index + 1}; 
                if ~islogical(plot_layout)
                    error('Optional input ''PlotLayout'' must be Boolean');
                end      
            case 'PlotPML'
                plot_PML = varargin{input_index + 1};
                if ~islogical(plot_PML)
                    error('Optional input ''PlotPML'' must be Boolean');
                end                 
            case 'PlotScale'
                plot_scale = varargin{input_index + 1};
                if ~strcmp(plot_scale, 'auto') && (~(numel(plot_scale) == 2 && isnumeric(plot_scale)))
                    error('Optional input ''PlotScale'' must be a 2 element numerical array or set to ''auto''');
                end                 
            case 'PlotSim'
                plot_sim = varargin{input_index + 1};
                if ~islogical(plot_sim)
                    error('Optional input ''PlotSim'' must be Boolean');
                end      
            case 'PMLAlpha'
                if length(varargin{input_index + 1}) > kgrid.dim
                    if kgrid.dim > 1
                        error(['Optional input ''PMLAlpha'' must be a 1 or ' kgrid.dim ' element numerical array']);
                    else
                        error('Optional input ''PMLAlpha'' must be a single numerical value');
                    end                    
                end
                switch kgrid.dim
                    case 1
                        PML_x_alpha = varargin{input_index + 1}(1); 
                    case 2
                        PML_x_alpha = varargin{input_index + 1}(1);
                        PML_y_alpha = varargin{input_index + 1}(end);                        
                    case 3
                        PML_x_alpha = varargin{input_index + 1}(1);
                        PML_y_alpha = varargin{input_index + 1}(ceil((end + 1)/2));
                        PML_z_alpha = varargin{input_index + 1}(end);
                end
            case 'PMLInside'
                PML_inside = varargin{input_index + 1};   
                if ~islogical(PML_inside)
                    error('Optional input ''PMLInside'' must be Boolean');
                end
            case 'PMLSize'
                if length(varargin{input_index + 1}) > kgrid.dim
                    if kgrid.dim > 1
                        error(['Optional input ''PMLSize'' must be a 1 or ' kgrid.dim ' element numerical array']);
                    else
                        error('Optional input ''PMLSize'' must be a single numerical value');
                    end
                end
                switch kgrid.dim
                    case 1
                        PML_x_size = round(varargin{input_index + 1}(1));
                    case 2
                        PML_x_size = round(varargin{input_index + 1}(1));
                        PML_y_size = round(varargin{input_index + 1}(end));
                    case 3
                        PML_x_size = round(varargin{input_index + 1}(1));
                        PML_y_size = round(varargin{input_index + 1}(ceil((end + 1)/2)));
                        PML_z_size = round(varargin{input_index + 1}(end));
                end
            case 'RecordMovie'
                record_movie = varargin{input_index + 1};    
                if ~islogical(record_movie)
                    error('Optional input ''RecordMovie'' must be Boolean');
                end
            case 'ReturnVelocity'
                return_velocity = varargin{input_index + 1};
                if ~islogical(return_velocity)
                    error('Optional input ''ReturnVelocity'' must be Boolean');
                end
            case 'StreamToDisk'
                if kgrid.dim == 3
                    stream_to_disk = varargin{input_index + 1};
                    if ~(numel(stream_to_disk) == 1 && ( isnumeric(stream_to_disk) || islogical(stream_to_disk) ))
                        error('Optional input ''StreamToDisk'' must be a single scalar or Boolean value');
                    end
                    
                    % if given as a Boolean, replace with the default
                    % number of time steps
                    if islogical(stream_to_disk) && (stream_to_disk ~= false)
                        stream_to_disk = STREAM_TO_DISK_STEPS_DEF;
                    end
                else
                    error('Optional input ''StreamToDisk'' is currently only compatible with 3D simulations');
                end
            case 'SaveToDisk'
                if kgrid.dim == 3
                    save_to_disk = varargin{input_index + 1};
                    if ~(islogical(save_to_disk) || ischar(save_to_disk))
                        error('Optional input ''SaveToDisk'' must be Boolean or a String');
                    end
                else
                    error('Optional input ''SaveToDisk'' is currently only compatible with 3D simulations');
                end
                if islogical(save_to_disk) && save_to_disk
                    save_to_disk = SAVE_TO_DISK_FILENAME_DEF;
                end
            case 'SaveToDiskExit'
                if kgrid.dim == 3
                    save_to_disk_exit = varargin{input_index + 1};
                    if ~islogical(save_to_disk_exit)
                        error('Optional input ''SaveToDiskExit'' must be Boolean');
                    end
                else
                    error('Optional input ''SaveToDiskExit'' is currently only compatible with 3D simulations');
                end  
            case 'Smooth'
                if length(varargin{input_index + 1}) > 3 || ~islogical(varargin{input_index + 1})
                    error('Optional input ''Smooth'' must be a 1, 2 or 3 element Boolean array');
                end
                smooth_p0 = varargin{input_index + 1}(1);
                smooth_c = varargin{input_index + 1}(ceil((end + 1)/2));
                smooth_rho0 = varargin{input_index + 1}(end);   
            case 'UsekSpace'
                use_kspace = varargin{input_index + 1}; 
                if ~islogical(use_kspace)
                    error('Optional input ''UsekSpace'' must be Boolean');
                end
            case 'UseSG'
                use_sg = varargin{input_index + 1}; 
                if ~islogical(use_sg)
                    error('Optional input ''UseSG'' must be Boolean');
                end                   
            otherwise
                error(['Unknown optional input ' varargin{input_index}]);
        end
    end
end

% =========================================================================
% CHECK FOR VALID INPUT COMBINATIONS
% =========================================================================

% enforce density input if velocity sources or output are being used
if ~user_medium_density_input && (ux_source || uy_source || uz_source || return_velocity)
    error('medium.density must be explicitly defined if velocity inputs or outputs are used, even in homogeneous media');
end

% enforce density input if nonlinear equations are being used
if ~user_medium_density_input && nonlinear
    error('medium.density must be explicitly defined if medium.BonA is specified');
end

% don't allow sensor.record_mode = 'statistics' with 'ReturnVelocity' set
% to true or Cartesian sensor masks
if use_sensor && ~store_time_series && (return_velocity || ~binary_sensor_mask)
    error('The optional input sensor.record_mode set to ''statistics'' is not currently compatible with Cartesian sensor masks or ''ReturnVelocity'' set to true.');
end

% check sensor compatability options for compute_directivity
if use_sensor && kgrid.dim == 2 && compute_directivity && ~binary_sensor_mask && strcmp(cartesian_interp, 'linear')
    error('sensor directivity fields are only compatible with binary sensor masks or ''CartInterp'' set to ''nearest''');
end   

% check input options for data streaming
if stream_to_disk && (~use_sensor || return_velocity || ~store_time_series || time_rev)
    error('The optional input ''StreamToDisk'' is currently only compatible with ''ReturnVelocity'' set to false (the default), sensor.record_mode set to ''time_history'' (the default), and forward simulations using a non-zero sensor mask.');
end

% switch off layout plot in time reversal mode
plot_layout = plot_layout && ~time_rev;

% check for automatic plot scaling
if strcmp(plot_scale, 'auto') 
    plot_scale_auto = true;
else
    plot_scale_auto = false;
end

% check for log plot scaling and store the plot scales
if plot_scale_log && ~plot_scale_auto
    alt_plot_scale_lin = plot_scale;
    alt_plot_scale_log = log10(abs(plot_scale) + log_scale_comp_factor) - log10(log_scale_comp_factor);
    alt_plot_scale_log(1) = -alt_plot_scale_log(1);
end

% force visualisation if record_movie is true
if record_movie
    plot_sim = true;
end

% ensure p0 smoothing is switched off if p0 is empty
if ~isfield(source, 'p0')
    smooth_p0 = false;
end

% ensure default display mask is switched off if sensor input is empty
if ~use_sensor && strcmp(display_mask, 'default')
    display_mask = 'off';
end

% switch off default display mask if using mesh plot
if kgrid.dim == 2 && mesh_plot && strcmp(display_mask, 'default')
    display_mask = 'off';
end

% start log if required
if create_log
    diary([LOG_NAME '.txt']);
end

% update command line status
if time_rev
    disp('  time reversal mode');
end

% check plot scaling if p0 is given
if isfield(source, 'p0') && ~time_rev && plot_sim && ~plot_scale_auto
    
    % find the maximum input pressure amplitude
    if isfield(source, 'p')
        max_val = max([source.p0(:); source.p(:)]);
    else
        max_val = max(source.p0(:));
    end
    
    % check the plot scaling
    if max_val > PLOT_SCALE_WARNING*plot_scale(2) || PLOT_SCALE_WARNING*max_val < plot_scale(2)
        disp('  WARNING: visualisation plot scale may not be optimal for given source');
    end
    
    clear max_val;
    
end

% cleanup unused variables
clear *_DEF NUM_REQ_INPUT_VARIABLES user_medium_density_input;