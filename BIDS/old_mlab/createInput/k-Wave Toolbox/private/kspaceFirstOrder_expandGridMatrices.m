% DESCRIPTION:
%       script to expand the grid matrices used in kspaceFirstOrder1D,
%       kspaceFirstOrder2D, and kspaceFirstOrder3D
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 20th August 2010
%       last update - 10th December 2011
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

% update the data type in case adding the PML requires additional index
% precision
if kgrid.total_grid_points < intmax('uint8');
    index_data_type = 'uint8';
elseif kgrid.total_grid_points < intmax('uint16');
    index_data_type = 'uint16';
elseif kgrid.total_grid_points < intmax('uint32');
    index_data_type = 'uint32';                
else
    index_data_type = 'double';
end  

% enlarge the sensor mask (for Cartesian sensor mask, this has already been
% converted to a binary mask for display in inputChecking)
if use_sensor
    if strcmp(class(sensor), 'kWaveTransducer')
        sensor.expand_grid(expand_size);
    else
        sensor.mask = expandMatrix(sensor.mask, expand_size, 0);
    end 
end

% enlarge the grid of sound speed by extending the edge values into the
% expanded grid 
if numel(c) > 1
    c = expandMatrix(c, expand_size);
end

% enlarge the grid of density by extending the edge values into the
% expanded grid    
if numel(rho0) > 1
    rho0 = expandMatrix(rho0, expand_size);
end
% keyboard
% enlarge the grid of medium.alpha_coeff if given
if isfield(medium, 'alpha_coeff') && numel(medium.alpha_coeff) > 1
    medium.alpha_coeff = expandMatrix(medium.alpha_coeff, expand_size);
end

% enlarge the grid of medium.BonA if given
if isfield(medium, 'BonA') && numel(medium.BonA) > 1
    medium.BonA = expandMatrix(medium.BonA, expand_size);
end

% enlarge the display mask if given
if ~(strcmp(display_mask, 'default') || strcmp(display_mask, 'off'))
    display_mask = expandMatrix(display_mask, expand_size, 0);
end

% enlarge the initial pressure if given
if isfield(source, 'p0')
    source.p0 = expandMatrix(source.p0, expand_size, 0);
end      

% enlarge the absorption filter mask if given
if isfield(medium, 'alpha_filter');
    medium.alpha_filter = expandMatrix(medium.alpha_filter, expand_size, 0);
end 

% enlarge the pressure source mask if given
if p_source   
    
    % enlarge the pressure source mask
    source.p_mask = expandMatrix(source.p_mask, expand_size, 0);

    % create an indexing variable corresponding to the source elements
    ps_index = find(source.p_mask ~= 0);
    
    % convert the data type depending on the number of indices
    eval(['ps_index = ' index_data_type '(ps_index);']);      

end

% enlarge the velocity source mask if given
if (ux_source || uy_source || uz_source || transducer_source)
    
    % update the source indexing variable
    if strcmp(class(source), 'kWaveTransducer')
        
        % check if the sensor is also the same transducer, if so, don't
        % expand the grid again
        if ~(strcmp(class(sensor), 'kWaveTransducer') && isequal(sensor, source))
        
            % expand the transducer mask
            source.expand_grid(expand_size);
            
        end
        
        % get the new active elements mask
        active_elements_mask = source.active_elements_mask;

        % update the indexing variable corresponding to the active elements
        us_index = find(active_elements_mask ~= 0);
        
        % clean up unused variables
        clear active_elements_mask;        
        
    else
        % enlarge the velocity source mask
        source.u_mask = expandMatrix(source.u_mask, expand_size, 0);
        
        % create an indexing variable corresponding to the source elements
        us_index = find(source.u_mask ~= 0);  
    end
    
    % convert the data type depending on the number of indices
    eval(['us_index = ' index_data_type '(us_index);']);   
    
end

% enlarge the directivity angle if given (2D only)
if use_sensor && kgrid.dim == 2 && compute_directivity
    sensor.directivity_angle = expandMatrix(sensor.directivity_angle, expand_size, 0);
end    