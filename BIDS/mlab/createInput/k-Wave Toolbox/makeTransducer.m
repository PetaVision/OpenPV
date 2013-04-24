function transducer = makeTransducer(kgrid, transducer_properties)
%MAKETRANSDUCER     Create k-Wave ultrasound transducer
%
% DESCRIPTION:
%       makeTransducer creates an object of the kWaveTransducer class which
%       can be substituted for the source or sensor inputs when using
%       kspaceFirstOrder3D.  
%
%       Note: This function will not work with older versions of MATLAB in
%       which custom class definitions are not supported. 
%
% USAGE:
%       transducer = makeTransducer(kgrid, settings)
%
% INPUTS:
%       
%       kgrid       - k-Wave grid structure returned by makeGrid
%       settings    - input structure used to define the properties of the
%                     transducer (see below) 
%
%       The following parameters can be appended as fields to the settings
%       input structure. These parameters are fixed when the transducer is
%       initialised (they cannot be changed without re-creating the
%       transducer). All the settings are optional and are given their
%       default values if not defined.    
%
%       number_elements - total number of transducer elements (default =
%                         128)
%       element_width   - width of each element in grid points
%                         (default = 1)
%       element_length  - length of each element in grid points
%                         (default = 10)
%       element_spacing - spacing (kerf width) between the transducer
%                         elements in grid points (default = 0)
%       position        - position of the corner of the transducer within
%                         the grid in grid points (default = [1, 1, 1])
%       radius          - radius of curvature of the transducer [m]
%                         (currently only inf is supported) (default = inf) 
%       input_signal    - signal used to drive the ultrasound transducer
%                         (where the sampling rate is defined by kgrid.dt
%                         or kgrid.t_array) (default = [])
%
%       The following parameters can also be appended as fields to the
%       settings input structure, and can additionally be modified after
%       the transducer has been initialised.
% 
%       active_elements - transducer elements that are currently active
%                         elements (default = all elements)
%       elevation_focus_distance - focus depth in the elevation direction
%                         [m] (default = inf) 
%       transmit_apodization - transmit apodization; can be set to any of
%                         the window shapes supported by getWin, or given
%                         as a vector the same length as number_elements
%                         (default = 'Rectangular')   
%       receive_apodization - receive apodization; can be set to any of the
%                         window shapes supported by getWin, or given as a
%                         vector the same length as number_elements
%                         (default = 'Rectangular')   
%       sound_speed     - sound speed used to calculate beamforming delays
%                         [m/s] (default = 1540) 
%       focus_distance  - focus distance used to calculate beamforming
%                         delays [m] (default = inf) 
%       steering_angle  - steering angle used to calculate beamforming
%                         delays [deg] (default = 0)
%       beamforming_delay_offset - beamforming delay offset (used to force
%                         beamforming delays to be positive) (default =
%                         'auto')  
%
% OUTPUTS:
%
%       transducer      - kWaveTransducer object which can be used to
%       replace the source or sensor inputs of kspaceFirstOrder3D 
% 
%       In addition to the input parameters given above (which are also
%       accessible after the transducer has been created), the
%       kWaveTransducer object has a number dependent properties and
%       methods.  
%
%       mask            - binary mask of the active transducer elements
%       transducer_width - total width of the transducer in grid points
%       number_active_elements - current number of active transducer
%                         elements 
%       input_signal    - user defined input signal appended and prepended
%                         with additional zeros depending on the values of
%                         focus_distance, elevation_focus_distance, and
%                         steering_angle  
%
%       all_elements_mask - return a binary mask of all the transducer
%                         elements (both active and inactive) 
%       active_elements_mask - return a binary mask of the active
%                         transducer elements (identical to mask) 
%       indexed_elements_mask - return a mask of all the transducer
%                         elements (both active and inactive), where the
%                         mask values indicate which transducer element
%                         each grid point corresponds to   
%       indexed_active_elements_mask - return a mask of the active
%                         transducer elements, where the mask values
%                         indicate which transducer element each grid point
%                         corresponds to   
%       beamforming_delays - return a vector of the beam forming delays (in
%                         units of time samples) for each active element
%                         based on the focus and steering angle settings 
%       elevation_beamforming_delays - return a vector of the elevation
%                         beam forming delays (in units of time samples)
%                         for each active element based on the elevation
%                         focus setting
%       delay_mask      - return a mask of the active transducer elements,
%                         where the mask values contain the beamforming
%                         delays (an integer input can also be given to
%                         control the beamforming delays used, where 1:
%                         both delays, 2: elevation only, 3: azimuth only)
%       transmit_apodization_mask - return a mask of the active transducer
%                         elements, where the mask values contain the
%                         apodization weights  
%       plot            - plot the transducer using voxelPlot
%       expand_grid     - increase the size of the computational grid to
%                         accomodate an external PML 
%       retract_grid    - reduce the size of the computational grid after
%                         simulation 
%       grid_size        - return the size of the computational grid
%       get_transmit_apodization - return the transmit apodization
%       get_receive_apodization - return the receive apodization
%       properties      - print a list of the transducer properties to the command line
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 28th July 2011
%       last update - 5th December 2011
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also makeGrid, kspaceFirstOrder3D, kWaveTransducer

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

try
    % create a new instance of the kWaveTransducer class
    transducer = kWaveTransducer(kgrid, transducer_properties);
catch ME
    % if user defined classes aren't supported, throw an error 
    if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
        error('The transducer cannot be created because user defined classes are not supported in your version of MATLAB. To use this functionality, please try using a newer MATLAB version.');
    end
    rethrow(ME);
end