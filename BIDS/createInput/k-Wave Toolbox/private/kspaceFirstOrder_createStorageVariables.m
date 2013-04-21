% DESCRIPTION:
%       subscript to create storage variables
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 1st August 2011
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

% =========================================================================
% PREPARE DATA MASKS AND STORAGE VARIABLES
% =========================================================================

if use_sensor

    % check sensor mask based on the Cartesian interpolation setting
    if ~binary_sensor_mask && strcmp(cartesian_interp, 'nearest')
        
        % extract the data using the binary sensor mask created in
        % inputChecking, but switch on Cartesian reorder flag so that the
        % final data is returned in the correct order (not in time
        % reversal mode).
        binary_sensor_mask = true;
        if ~time_rev
            reorder_data = true;
        end

        % check if any duplicate points have been discarded in the
        % conversion from a Cartesian to binary mask
        num_discarded_points = length(sensor_x) - sum(sensor.mask(:));
        if num_discarded_points ~= 0
            disp(['  WARNING: ' num2str(num_discarded_points) ' duplicated sensor points discarded (nearest neighbour interpolation)']);
        end        

    end
    
    % create mask indices (this works for both normal sensor and transducer
    % inputs)
    sensor_mask_ind = find(sensor.mask ~= 0);
    
    % convert the data type depending on the number of indices (this saves
    % memory)
    eval(['sensor_mask_ind = ' index_data_type '(sensor_mask_ind);']); 

    % create storage and scaling variables
    if ~time_rev
        
        % preallocate storage variables, if return_velocity is true, the
        % outputs are assigned as structure fields, otherwise the
        % appropriate pressure data is assigned directly to sensor_data
        if return_velocity
            
            % pre-allocate the sensor_data variable based on the number of
            % binary or Cartesian sensor points
            if binary_sensor_mask
                sensor_data.p = zeros(sum(sensor.mask(:)), length(t_array));
            else
                sensor_data.p = zeros(length(sensor_x), length(t_array));
            end
            
            % pre-allocate the velocity fields based on the number of
            % dimensions in the simulation
            switch kgrid.dim
                case 1
                    sensor_data.ux = sensor_data.p;
                case 2
                    sensor_data.ux = sensor_data.p;
                    sensor_data.uy = sensor_data.p;
                case 3
                    sensor_data.ux = sensor_data.p;
                    sensor_data.uy = sensor_data.p;
                    sensor_data.uz = sensor_data.p;  
            end
        
        % return velocity is false    
        else
            
            % if streaming data to disk, reduce to the size of the
            % sensor_data matrix based on the value of stream_to_disk
            if kgrid.dim == 3 && stream_to_disk
                num_time_points = stream_to_disk;
                
                % initialise the file index variable
                stream_data_index = 1;
            else
                num_time_points = length(t_array);
            end
            
            % binary sensor mask
            if binary_sensor_mask
                if transducer_sensor
                    if transducer_receive_elevation_focus
                        % if there is elevation focusing, a buffer is
                        % needed to store a short time history at each
                        % sensor point before averaging
                        sensor_data_buffer_size = max(sensor.elevation_beamforming_delays) + 1;
                        if sensor_data_buffer_size > 1
                            sensor_data_buffer = zeros(sum(sensor.mask(:)), sensor_data_buffer_size);
                        else
                            clear sensor_data_buffer sensor_data_buffer_size;
                            transducer_receive_elevation_focus = false;
                        end
                   end
                    
                    % the grid points can be summed on the fly and so the
                    % sensor is the size of the number of active elements 
                    sensor_data = zeros(sensor.number_active_elements, num_time_points);
                    
                else
                    if store_time_series
                        % store time series
                        sensor_data = zeros(sum(sensor.mask(:)), num_time_points);
                    else
                        % store only statistics at each sensor point
                        sensor_data.p_rms = zeros(sum(sensor.mask(:)), 1);
                        sensor_data.p_max = zeros(sum(sensor.mask(:)), 1);
                    end
                end
                
            % Cartesian sensor mask    
            else
                if store_time_series
                    % store time series
                    sensor_data = zeros(length(sensor_x), num_time_points);
                else
                    % store only statistics at each sensor point
                    sensor_data.p_rms = zeros(length(sensor_x), 1);
                    sensor_data.p_max = zeros(length(sensor_x), 1);
                end
                
            end
            
            clear num_time_points;
        end       
    end
end

% =========================================================================
% PRECOMPUTE DATA STORAGE CASE
% =========================================================================

if time_rev || ~use_sensor
    % do not store any data in time reversal mode
    extract_data_case = 0;
elseif ~(kgrid.dim == 2 && compute_directivity)
    if ~return_velocity && ~binary_sensor_mask
        % return velocity = false
        % binary sensor mask = false
        extract_data_case = 1;
        
    elseif ~return_velocity && binary_sensor_mask
        % return velocity = false
        % binary sensor mask = true      
        extract_data_case = 2;

    elseif return_velocity && ~binary_sensor_mask 
        % return velocity = true
        % binary sensor mask = false     
        extract_data_case = 3;

    elseif return_velocity && binary_sensor_mask 
        % return velocity = true
        % binary sensor mask = true    
        extract_data_case = 4;

    else
        error('Unknown data output combination...');
    end
elseif ~return_velocity && binary_sensor_mask 
    % compute directivity = true (only supported in 2D)
    % return velocity = false (must be false if directivity = true)
    % binary sensor mask = true (must be true if directivity = true)
    extract_data_case = 5;      
else
    error('Unknown data output combination...');    
end

% =========================================================================
% CHECK SUPPORTED CARTESIAN SENSOR MASK INTERPOLATION OPTIONS
% =========================================================================

% precomputate the triangulation points if a Cartesian sensor mask is
% used with linear interpolation
if use_sensor && ~time_rev && ~binary_sensor_mask
    if kgrid.dim == 2
        % check if data casting is used
        if ~FORCE_TSEARCH && strcmp(data_cast, 'off');
            % try to use TriScatteredInterp (only supported post R2009a)
            % NOTE: this function only supports double precision numbers so
            % cannot be used in conjunction with 'DataCast' options
            try
                use_TriScatteredInterp = true;
                disp('  calculating Delaunay triangulation (TriScatteredInterp)...');
                F_interp = TriScatteredInterp(reshape(kgrid.x, [], 1), reshape(kgrid.y, [], 1), reshape(zeros(kgrid.Nx, kgrid.Ny), [], 1));
            catch ME
                % if TriScatteredInterp doesn't exist, use gridDataFast 
                if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
                    disp('  TriScatteredInterp not supported');
                    disp('  calculating Delaunay triangulation (tsearch)...')
                    [zi, del_tri] = gridDataFast(kgrid.x, kgrid.y, zeros(kgrid.Nx, kgrid.Ny), sensor_x, sensor_y);
                    use_TriScatteredInterp = false;
                else
                    rethrow(ME);
                end
            end
        else
            % try to use gridDataFast
            try
                disp('  calculating Delaunay triangulation (tsearch)...')
                [zi, del_tri] = gridDataFast(kgrid.x, kgrid.y, zeros(kgrid.Nx, kgrid.Ny), sensor_x, sensor_y);
                use_TriScatteredInterp = false;
            catch ME
                error('The inbuilt function tsearch is not supported in this version of MATLAB. Try running the simulation with ''CartInterp'' set to ''nearest'' or using a different data type.');
            end
        end   
    elseif kgrid.dim == 3
        % try to use TriScatteredInterp (only supported post R2009a)
        % NOTE: this function only supports double precision numbers so
        % cannot be used in conjunction with 'DataCast' options
        try
            disp('  calculating Delaunay triangulation (TriScatteredInterp)...');
            F_interp = TriScatteredInterp(reshape(kgrid.x, [], 1), reshape(kgrid.y, [], 1), reshape(kgrid.z, [], 1), reshape(zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz), [], 1));
        catch ME
            % if TriScatteredInterp doesn't exist, return an error
            if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
                error('The inbuilt function TriScatteredInterp is not supported in this version of MATLAB. Try running the simulation with ''CartInterp'' set to ''nearest''');
            else
                rethrow(ME);
            end
        end  
    end
end

% % For MATLAB versions prior to 2008a, comment out lines 197 - 247 above
% % and uncomment lines 249 - 263 below (tested using MATLAB 2007a).
% if use_sensor && ~time_rev && ~binary_sensor_mask
%     if kgrid.dim == 2
%         try
%             disp('  calculating Delaunay triangulation (tsearch)...')
%             [zi, del_tri] = gridDataFast(kgrid.x, kgrid.y, zeros(kgrid.Nx, kgrid.Ny), sensor_x, sensor_y);
%             use_TriScatteredInterp = false;
%         catch
%             error('The inbuilt function tsearch is not supported in this version of MATLAB. Try running the simulation with ''CartInterp'' set to ''nearest'' or using a different data type');
%         end
%     elseif kgrid.dim == 3
%         error('3D linear interpolation is not supported in this version of MATLAB. Try running the simulation with ''CartInterp'' set to ''nearest''');
%     end
% end