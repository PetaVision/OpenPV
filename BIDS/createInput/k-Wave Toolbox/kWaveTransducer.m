%KWAVETRANSDUCER   Class definition for k-Wave transducer
%
% DESCRIPTION:
%       See makeTransducer for function arguments.
%
% ABOUT:
%       author          - Bradley Treeby
%       date            - 9th December 2010
%       last update     - 5th December 2011
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

classdef kWaveTransducer < handle

    % define the properties of the transducer that cannot be modified by
    % the user after the transducer is initialised (these parameters are
    % stored). The numbers assigned here are the default values used if the
    % parameters are not set explicitly by the user.
    properties (GetAccess = 'public', SetAccess = 'private')
        
        % the total number of transducer elements
        number_elements = 128;        
        
        % the width of each element in grid points
        element_width = 1;
        
        % the length of each element in grid points
        element_length = 20;
        
        % the spacing (kerf width) between the transducer elements in grid
        % points
        element_spacing = 0;    
        
        % the position of the corner of the transducer in the grid
        position = [1, 1, 1];      
        
        % the radius of curvature of the transducer [m]
        radius = inf;        
        
    end
    
    % define the properties of the transducer that can be modified by the
    % user after the transducer is initialised (these parameters are
    % stored). The numbers assigned here are the default values used if the
    % parameters are not set explicitly by the user.
    properties (Access = 'public')
        
        % the transducer elements that are currently active elements
        active_elements;        
               
        % the focus depth in the elevation direction [m]
        elevation_focus_distance = inf;

        % transmit apodization
        transmit_apodization = 'Rectangular';        
        
        % receive apodization
        receive_apodization = 'Rectangular';
        
        % sound speed used to calculate beamforming delays [m/s]
        sound_speed = 1540;
        
        % focus distance used to calculate beamforming delays [m]
        focus_distance = inf;
        
        % steering angle used to calculate beamforming delays [deg]
        steering_angle = 0;
        
        % beamforming delay offset (used to force beamforming delays to be
        % positive)
        beamforming_delay_offset = 'auto';
        
    end
    
    % define the hidden properties (these cannot be seen or accessed
    % directly by the user) 
    properties (Hidden = true, Access = 'private')
        
        % indexed sensor mask from which the other masks are computed
        indexed_mask;
        
        % indexed mask for the voxels in each element used to compute
        % elevation focussing
        indexed_element_voxel_mask;
        
        % size of the grid in which the transducer is defined
        stored_grid_size;
        
        % corresponding grid spacing
        grid_spacing; 
        
        % time spacing
        dt;

        % original copy of the user defined input signal
        stored_input_signal;
        
    end
    
    % define the dependent properties (these parameters are computed when
    % queried) 
    properties(Dependent = true)
        
        % binary mask of the active elements (this returns
        % obj.active_elements_mask). The additional name is to allow
        % compatability when the transducer is used in place of sensor or
        % source.
        mask;

        % total width of the transducer in grid points
        transducer_width;
        
        % current number of active transducer elements
        number_active_elements;
        
        % user defined input signal with appended zeros dependent on the
        % beamforming (steering_angle, focus_distance) settings
        input_signal;
        
    end
    
    % constructor function
    methods
        function transducer = kWaveTransducer(kgrid, transducer_properties)
            
            % allocate the grid size and spacing
            transducer.stored_grid_size = [kgrid.Nx, kgrid.Ny, kgrid.Nz];
            transducer.grid_spacing = [kgrid.dx, kgrid.dy, kgrid.dz];
            
            % allocate the temporal spacing
            if isnumeric(kgrid.dt)
                transducer.dt = kgrid.dt;
            elseif isnumeric(kgrid.t_array)
                transducer.dt = kgrid.t_array(2) - kgrid.t_array(1);
            else
                error('kgrid.dt or kgrid.t_array must be explicitly defined');
            end
            
            % check for properties input
            if nargin == 1
                transducer_properties = [];
            end
            
            % check the input fields
            checkFieldNames(transducer_properties, {'number_elements', 'active_elements', 'element_width',...
                'element_length', 'element_spacing', 'elevation_focus_distance', 'position', 'radius', 'receive_apodization',...
                'transmit_apodization', 'sound_speed', 'focus_distance', 'steering_angle', 'input_signal'});
            
            % replace default settings with user defined properties
            % -------------------------------------------------------------
            
            if isfield(transducer_properties, 'number_elements')
                % force value to be a positive integer
                transducer.number_elements = round(abs(transducer_properties.number_elements));
            end
            
            if isfield(transducer_properties, 'active_elements')
                transducer.active_elements = transducer_properties.active_elements;
            else
                transducer.active_elements = ones(transducer_properties.number_elements, 1);
            end
    
            if isfield(transducer_properties, 'element_width')
                % force value to be a positive integer
                transducer.element_width = round(abs(transducer_properties.element_width));
            end
            
            if isfield(transducer_properties, 'element_length')
                % force value to be a positive integer
                transducer.element_length = round(abs(transducer_properties.element_length));
            end
            
            if isfield(transducer_properties, 'element_spacing')
                % force value to be a positive integer
                transducer.element_spacing = round(abs(transducer_properties.element_spacing));
            end
            
            if isfield(transducer_properties, 'elevation_focus_distance')
                transducer.elevation_focus_distance = transducer_properties.elevation_focus_distance;
            end
            
            if isfield(transducer_properties, 'position')
                % force values to be positive integers
                transducer.position = round(abs(transducer_properties.position));
            end
            
            if isfield(transducer_properties, 'radius')
                transducer.radius = transducer_properties.radius;
                
                % only allow an infinite radius for now
                if ~isinf(transducer.radius)
                    error('Only a value of transducer.radius = inf is currently supported');
                end
            end
            
            if isfield(transducer_properties, 'receive_apodization')
                % if a user defined apodization is given explicitly, check
                % the length of the input
                if isnumeric(transducer_properties.receive_apodization) ...
                        && (length(transducer_properties.receive_apodization) ~= transducer.number_active_elements)
                    error('The length of the receive apodization input must match the number of active elements');
                end
                                       
                % assign the input                  
                transducer.receive_apodization = transducer_properties.receive_apodization;            
            end
            
            if isfield(transducer_properties, 'transmit_apodization')
                % if a user defined apodization is given explicitly, check
                % the length of the input
                if isnumeric(transducer_properties.transmit_apodization) ...
                        && (length(transducer_properties.transmit_apodization) ~= transducer.number_active_elements)
                    error('The length of the transmit apodization input must match the number of active elements');
                end
                    
                % assign the input
                transducer.transmit_apodization = transducer_properties.transmit_apodization;              
            end            
                                
            if isfield(transducer_properties, 'sound_speed')
                transducer.sound_speed = transducer_properties.sound_speed;
                
                % check to see the sound_speed is positive
                if ~(transducer.sound_speed> 0)
                    error('transducer.sound_speed must be greater than 0');
                end                 
            end
            
            if isfield(transducer_properties, 'focus_distance')
                transducer.focus_distance = transducer_properties.focus_distance;
            end
            
            if isfield(transducer_properties, 'steering_angle')
                transducer.steering_angle = transducer_properties.steering_angle;
                
                % check if the steering angle is between -90 and 90
                if ~((transducer.steering_angle > -90) && (transducer.steering_angle < 90))
                    error('transducer.steering_angle must be betweeb -90 and 90 degrees');
                end
            end
            
            if isfield(transducer_properties, 'input_signal')
                transducer.stored_input_signal = transducer_properties.input_signal;
                
                % force the input signal to be a column vector
                if numDim(transducer.input_signal) == 1
                    transducer.stored_input_signal = reshape(transducer.stored_input_signal, [], 1);
                else
                    error('transducer.input_signal must be a one-dimensional array');
                end
            end            
            
            % -------------------------------------------------------------
            
            % check the transducer fits into the grid
            if sum(transducer.position(:) == 0)
                error('The defined transducer must be positioned within the grid');
            elseif (transducer.position(2) + transducer.number_elements*transducer.element_width + (transducer.number_elements - 1)*transducer.element_spacing) > transducer.stored_grid_size(2)
                error('The defined transducer is too large or positioned outside the grid in the y-direction');
            elseif (transducer.position(3) + transducer.element_length) > transducer.stored_grid_size(3)
                transducer.position(3)
                transducer.element_length
                transducer.stored_grid_size(3)
                error('The defined transducer is too large or positioned outside the grid in the z-direction');
            elseif transducer.position(1) > transducer.stored_grid_size(1)
                error('The defined transducer is positioned outside the grid in the x-direction');
            end
         
            % assign the data type for the transducer matrix based on the
            % number of different elements (uint8 supports 255 numbers so
            % most of the time this data type will be used)
            if transducer.number_elements < intmax('uint8');
                mask_type = 'uint8';
            elseif transducer.number_elements < intmax('uint16');
                mask_type = 'uint16';
            elseif transducer.number_elements < intmax('uint32');
                mask_type = 'uint32';                
            else
                mask_type = 'uint64';
            end
            
            % create an empty transducer mask (the grid points within
            % element 'n' are all given the value 'n')
            transducer.indexed_mask = zeros(transducer.stored_grid_size, mask_type);
                
            % create a second empty mask used for the elevation beamforming
            % delays (the grid points across each element are numbered 1 to
            % M, where M is the number of grid points in the elevation
            % direction)
            transducer.indexed_element_voxel_mask = zeros(transducer.stored_grid_size, mask_type);
                
            % create the corresponding indexing variable 1:M
            element_voxel_index = repmat(1:transducer.element_length, transducer.element_width, 1);
                            
            % for each transducer element, calculate the grid point indices
            for element_index = 1:transducer.number_elements

                % assign the current transducer position
                element_pos_x = transducer.position(1);
                element_pos_y = transducer.position(2) + (transducer.element_width + transducer.element_spacing)*(element_index - 1);
                element_pos_z = transducer.position(3);
                
                % assign the grid points within the current element the
                % index of the element
                transducer.indexed_mask(element_pos_x, element_pos_y:element_pos_y + transducer.element_width - 1, element_pos_z:element_pos_z + transducer.element_length - 1) = element_index;
                
                % assign the individual grid points an index corresponding
                % to their order across the element
                transducer.indexed_element_voxel_mask(element_pos_x, element_pos_y:element_pos_y + transducer.element_width - 1, element_pos_z:element_pos_z + transducer.element_length - 1) = element_voxel_index;

            end

            % double check the transducer fits within the desired grid size
            if any(size(transducer.indexed_elements_mask) ~= transducer.stored_grid_size)
                error('Desired transducer is larger than the input grid_size');
            end
            
        end
    end

    % set and get functions for dependent variables that only run when queried
    methods
        
        % allow mask query to allow compatability with regular sensor
        % structure - return the active sensor mask
        function mask = get.mask(obj)
            mask = obj.active_elements_mask;
        end        
        
        % return input signal
        function signal = get.input_signal(obj)
            
            % copy the stored input signal
            signal = obj.stored_input_signal;
            
            % check the signal is not empty
            if isempty(signal)
                error('Transducer input signal is not defined');
            end
            
            % get the current delay beam forming 
            delay_mask = obj.delay_mask;
            
            % find the maximum delay
            delay_max = max(delay_mask(:));
            
            % count the number of leading zeros in the input signal
            leading_zeros = find(signal ~= 0, 1) - 1;

            % count the number of trailing zeros in the input signal
            trailing_zeros = find(flipud(signal) ~= 0, 1) - 1;
            
            % check the number of leading zeros is sufficient given the
            % maximum delay
            if leading_zeros < delay_max + 1
                disp(['  prepending transducer.input_signal with ' num2str(delay_max - leading_zeros + 1) ' leading zeros']);
                
                % prepend extra leading zeros
                signal = [zeros(delay_max - leading_zeros + 1, 1); signal];                
            end
                   
            % check the number of leading zeros is sufficient given the
            % maximum delay
            if trailing_zeros < delay_max + 1
                disp(['  appending transducer.input_signal with ' num2str(delay_max - trailing_zeros + 1) ' trailing zeros']);
                
                % append extra trailing zeros
                signal = [signal; zeros(delay_max - trailing_zeros + 1, 1)];
            end            
            
        end
        
        function transducer_width = get.transducer_width(obj)
            % return the overall length of the transducer
            transducer_width = obj.number_elements*obj.element_width + (obj.number_elements - 1)*obj.element_spacing;
        end
        
        function number_active_elements = get.number_active_elements(obj)
            % return the current number of active elements
            number_active_elements = sum(obj.active_elements(:));
        end     
        
    end
    
    % general class methods
    methods
        
        % return a binary mask showing the locations of all the elements
        % (both active and inactive)
        function mask = all_elements_mask(obj)
            mask = obj.indexed_mask;
            mask(mask ~= 0) = 1;
        end        
        
        % return a binary mask showing the locations of the active elements
        function mask = active_elements_mask(obj)
            
            % copy the indexed elements mask
            mask = obj.indexed_mask;
            
            % remove inactive elements from the mask
            for element_index = 1:obj.number_elements
                mask(mask == element_index) = obj.active_elements(element_index);
            end
            
            % convert remaining mask to binary
            mask(mask ~= 0) = 1;

        end      
        
        % return a copy of the indexed elements mask
        function mask = indexed_elements_mask(obj)
            mask = obj.indexed_mask;
        end
        
        % return a copy of the indexed elements mask only for the active
        % elements
        function mask = indexed_active_elements_mask(obj)
            
            % copy the indexed elements mask
            mask = obj.indexed_mask;
            
            % remove inactive elements from the mask
            for element_index = 1:obj.number_elements
                if ~(obj.active_elements(element_index))
                    mask(mask == element_index) = 0;
                end
            end
            
            % force the lowest element index to be 1
            lowest_active_element_index = find(obj.active_elements, 1);
            mask(mask ~= 0) = mask(mask ~= 0) - lowest_active_element_index + 1;

        end          
        
        % calculate the beamforming delays based on the focus and steering
        % settings
        function delay_times = beamforming_delays(obj)
            
            % calculate the element pitch in [m]
            element_pitch = (obj.element_spacing + obj.element_width)*obj.grid_spacing(2);

            % create indexing variable
            element_index = -(obj.number_active_elements - 1)/2:(obj.number_active_elements - 1)/2;           

            % check for focus depth
            if isinf(obj.focus_distance)
                % calculate time delays for a steered beam  
                delay_times = -element_pitch*element_index*sin(obj.steering_angle*pi/180)/(obj.sound_speed);        % [s]
            else   
                % calculate time delays for a steered and focussed beam
                delay_times = obj.focus_distance/obj.sound_speed * (1 - sqrt(...
                    1 ... 
                    + (element_index*element_pitch./obj.focus_distance).^2 ... 
                    - 2 * (element_index*element_pitch./obj.focus_distance) * sin(obj.steering_angle*pi/180) ...
                    ));      % [s]
            end
                        
            % convert the delays to be in units of time points and then
            % reverse
            delay_times = -round(delay_times/obj.dt);            
            
        end
        
        % calculate the elevation beamforming delays based on the focus 
        % setting
        function delay_times = elevation_beamforming_delays(obj)
            
            % create indexing variable
            element_index = -(obj.element_length - 1)/2:(obj.element_length - 1)/2; 

            % calculate time delays for a focussed beam
            delay_times = (obj.elevation_focus_distance - sqrt( (element_index*obj.grid_spacing(3)).^2 + obj.elevation_focus_distance^2 ))./obj.sound_speed;

            % convert the delays to be in units of time points and then
            % reverse 
            delay_times = -round(delay_times/obj.dt);

        end
        
        % calculate the elevation beamforming delays based on the focus 
        % setting
        function mask = elevation_beamforming_mask(obj)
            
            % get elevation beamforming mask
            delay_mask = obj.delay_mask(2);
            
            % extract the active elements
            delay_mask = delay_mask(obj.active_elements_mask ~= 0);
            
            % create an empty output mask
            mask = zeros(length(delay_mask), max(delay_mask) + 1);
            
            % populate the mask by setting 1's at the index given by the
            % delay time
            for index = 1:length(delay_mask)
                mask(index, delay_mask(index) + 1) = 1;
            end
            
            % flip the mask so the shortest delays are at the right
            mask = fliplr(mask);
        end
        
        % calculate the beamforming weights and return in the form of a
        % delay mask
        function mask = delay_mask(obj, mode)
            % mode == 1: both delays
            % mode == 2: elevation only
            % mode == 3: azimuth only
                                  
            % assign the delays to a new mask using the indexed_element_mask
            indexed_active_elements_mask = obj.indexed_active_elements_mask;
            active_elements_index = find(indexed_active_elements_mask ~= 0);
            mask = zeros(obj.stored_grid_size);
            
            % calculate azimuth focus delay times
            if ~isinf(obj.focus_distance) && (nargin == 1 || mode ~= 2)
                
                % get the element beamforming delays
                delay_times = obj.beamforming_delays;
                
                % add delay times
                mask(active_elements_index) = delay_times(indexed_active_elements_mask(active_elements_index));
                
            end

            % calculate elevation focus time delays
            if ~isinf(obj.elevation_focus_distance) && (nargin == 1 || mode ~= 3)
                
                % get elevation beamforming delays
                elevation_delay_times = obj.elevation_beamforming_delays;
                
                % get current mask
                element_voxel_mask = obj.indexed_element_voxel_mask;
                                                
                % add delay times
                mask(active_elements_index) = mask(active_elements_index) + elevation_delay_times(element_voxel_mask(active_elements_index)).';
                
            end
            
            % shift delay times (these should all be >= 0)
            if strcmp(obj.beamforming_delay_offset, 'auto')
                mask = mask - min(mask(:));            
            else
                mask = mask + obj.beamforming_delay_offset;
            end            
                        
        end
              
        % convert the transmit apodization into the form of a element mask
        function mask = transmit_apodization_mask(obj)
            
            % check if a user defined apodization is given and whether this
            % is still the correct size (in case the number of active
            % elements has changed)
            if isnumeric(obj.transmit_apodization)
                if (length(obj.transmit_apodization) ~= obj.number_active_elements)
                    error('The length of the transmit apodization input must match the number of active elements');
                else
                    % assign apodization
                    apodization = obj.transmit_apodization;
                end
            else
                % create apodization using getWin
                apodization = getWin(obj.number_active_elements, obj.transmit_apodization);
            end
            
            % create an empty mask;
            mask = zeros(obj.stored_grid_size);
            
            % assign the apodization values to every grid point in the
            % transducer
            mask_index = obj.indexed_active_elements_mask;
            mask_index = mask_index(mask_index ~= 0);
            mask(obj.active_elements_mask == 1) = apodization(mask_index);
            
        end
        
        % return the transmit apodization
        function apodization = get_transmit_apodization(obj)
            
            % check if a user defined apodization is given and whether this
            % is still the correct size (in case the number of active
            % elements has changed)
            if isnumeric(obj.transmit_apodization)
                if (length(obj.transmit_apodization) ~= obj.number_active_elements)
                    error('The length of the transmit apodization input must match the number of active elements');
                else
                    % assign apodization
                    apodization = obj.transmit_apodization;
                end
            else
                % create apodization using getWin
                apodization = getWin(obj.number_active_elements, obj.transmit_apodization);
            end           
        end                
        
        % return the receive apodization
        function apodization = get_receive_apodization(obj)
            
            % check if a user defined apodization is given and whether this
            % is still the correct size (in case the number of active
            % elements has changed)
            if isnumeric(obj.receive_apodization)
                if (length(obj.receive_apodization) ~= obj.number_active_elements)
                    error('The length of the receive apodization input must match the number of active elements');
                else
                    % assign apodization
                    apodization = obj.receive_apodization;
                end
            else
                % create apodization using getWin
                apodization = getWin(obj.number_active_elements, obj.receive_apodization);
            end           
        end        
        
        % function to create an a-line based on the input sensor data and
        % the current apodization and beamforming setting
        function line = scan_line(obj, sensor_data)
           
            % get the current apodization setting
            apodization = obj.get_receive_apodization;
            
            % get the current beamforming weights
            delays = obj.beamforming_delays;
            
            % offset the received sensor_data by the beamforming delays and
            % apply receive apodization 
            for element_index = 1:obj.number_active_elements
                sensor_data(element_index, :) = apodization(element_index).*[sensor_data(element_index, 1 + delays(element_index):end), zeros(1, delays(element_index))];
            end

            % form the a-line summing across the elements
            line = sum(sensor_data);
            
        end
        
        % plot the transducer using voxelPlot
        function plot(obj)
            voxelPlot(double(obj.all_elements_mask));
        end
        
        % allow the mask to be resized by the simulation functions
        function expand_grid(obj, expand_size)
            obj.indexed_mask = expandMatrix(obj.indexed_mask, expand_size, 0);
        end
        
        % allow the mask to be resized by the simulation functions
        function retract_grid(obj, retract_size)
            obj.indexed_mask = obj.indexed_mask(1 + retract_size(1):end - retract_size(1), 1 + retract_size(2):end - retract_size(2), 1 + retract_size(3):end - retract_size(3));
        end 
        
        % return the size of the grid
        function grid_size = grid_size(obj)
            grid_size = obj.stored_grid_size;
        end
        
        % print out the transducer properties
        function properties(obj)
            disp(' ');
            disp('k-Wave Transducer Properties');
            disp(['  transducer position: [' num2str(obj.position) ']']);
            disp(['  transducer width: ' scaleSI(obj.transducer_width*obj.grid_spacing(2)) 'm (' num2str(obj.transducer_width) ' grid points)']);
            disp(['  number of elements: ' num2str(obj.number_elements)]);
            disp(['  number of active elements: ' num2str(obj.number_active_elements) ' (elements ' num2str(find(obj.active_elements, 1, 'first')) ' to ' num2str(find(obj.active_elements, 1, 'last')) ')']);
            disp(['  element width: ' scaleSI(obj.element_width*obj.grid_spacing(2)) 'm (' num2str(obj.element_width) ' grid points)']);
            disp(['  element spacing (kerf): ' scaleSI(obj.element_spacing*obj.grid_spacing(2)) 'm (' num2str(obj.element_spacing) ' grid points)']);
            disp(['  element pitch: ' scaleSI((obj.element_spacing + obj.element_width)*obj.grid_spacing(2)) 'm (' num2str((obj.element_spacing + obj.element_width)) ' grid points)']);
            disp(['  element length: ' scaleSI(obj.element_length*obj.grid_spacing(3)) 'm (' num2str(obj.element_length) ' grid points)']);
            disp(['  sound speed: ' num2str(obj.sound_speed) 'm/s']);
            if isinf(obj.focus_distance)
                disp('  focus distance: infinite');
            else
                disp(['  focus distance: ' scaleSI(obj.focus_distance) 'm']);
            end
            if isinf(obj.elevation_focus_distance)
                disp('  elevation focus distance: infinite');
            else
                disp(['  elevation focus distance: ' scaleSI(obj.elevation_focus_distance) 'm']);
            end
            disp(['  steering angle: ' num2str(obj.steering_angle) ' degrees']);
        end
      
    end
end