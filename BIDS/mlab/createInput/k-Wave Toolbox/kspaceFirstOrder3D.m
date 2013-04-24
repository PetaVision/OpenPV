function [sensor_data, field_data, mem_usage] = kspaceFirstOrder3D(kgrid, medium, source, sensor, varargin)
%KSPACEFIRSTORDER3D     3D time-domain simulation of wave propagation.
%
% DESCRIPTION:
%       kspaceFirstOrder3D simulates the time-domain propagation of linear
%       compressional waves through a three-dimensional homogeneous or
%       heterogeneous acoustic medium given four input structures: kgrid,
%       medium, source, and sensor. The computation is based on a
%       first-order k-space model which accounts for power law absorption
%       and a heterogeneous sound speed and density. If medium.BonA is
%       specified, cumulative nonlinear effects are also modelled. At each
%       time-step (defined by kgrid.t_array), the pressure at the positions
%       defined by sensor.mask are recorded and stored. If kgrid.t_array is
%       set to 'auto', this array is automatically generated using
%       makeTime. An anisotropic absorbing boundary condition called a
%       perfectly matched layer (PML) is implemented to prevent waves that
%       leave one side of the domain being reintroduced from the opposite
%       side (a consequence of using the FFT to compute the spatial
%       derivatives in the wave equation). This allows infinite domain
%       simulations to be computed using small computational grids.
%
%       For a homogeneous medium the formulation is exact and the
%       time-steps are only limited by the effectiveness of the perfectly
%       matched layer. For a heterogeneous medium, the solution represents
%       a leap-frog pseudospectral method with a Laplacian correction that
%       improves the accuracy of computing the temporal derivatives. This
%       allows larger time-steps to be taken without instability compared
%       to conventional pseudospectral time-domain methods. The
%       computational grids are staggered both spatially and temporally.
%
%       An initial pressure distribution can be specified by assigning a
%       matrix (the same size as the computational grid) of arbitrary
%       numeric values to source.p0. A time varying pressure source can
%       similarly be specified by assigning a binary matrix (i.e., a matrix
%       of 1's and 0's with the same dimensions as the computational grid)
%       to source.p_mask where the 1's represent the grid points that form
%       part of the source. The time varying input signals are then
%       assigned to source.p. This must be the same length as kgrid.t_array
%       and can be a single time series (in which case it is applied to all
%       source elements), or a matrix of time series following the source
%       elements using MATLAB's standard column-wise linear matrix index
%       ordering. A time varying velocity source can be specified in an
%       analogous fashion, where the source location is specified by
%       source.u_mask, and the time varying input velocity is assigned to
%       source.ux, source.uy, and source.uz.
%
%       The pressure is returned as an array of time series at the sensor
%       locations defined by sensor.mask. This can be given either as a
%       binary grid (i.e., a matrix of 1's and 0's with the same dimensions
%       as the computational grid) representing the grid points within the
%       computational grid that will collect the data, or as a series of
%       arbitrary Cartesian coordinates within the grid at which the
%       pressure values are calculated at each time step via interpolation.
%       The Cartesian points must be given as a 3 by N matrix corresponding
%       to the x, y, and z positions, respectively. The final pressure
%       field over the complete computational grid can also be obtained
%       using the output field_data. If no output is required, the sensor
%       input can be replaced with an empty array []. Both the source and
%       sensor inputs can also be replaced by an object of the
%       kWaveTransducer class created using makeTransducer.
%
%       If sensor.mask is given as a set of Cartesian coordinates, the
%       computed sensor_data is returned in the same order. If sensor.mask
%       is given as a binary grid, sensor_data is returned using MATLAB's
%       standard column-wise linear matrix index ordering. In both cases,
%       the recorded data is indexed as sensor_data(sensor_position, time).
%       For a binary sensor mask, the pressure values at a particular time
%       can be restored to the sensor positions within the computation grid
%       using unmaskSensorData.
%
%       By default, the recorded pressure field is passed directly to the
%       output arguments sensor_data and field_data. However, the particle
%       velocity can also be recorded by setting the optional input
%       parameter 'ReturnVelocity' to true. In this case, the output
%       arguments sensor_data and field_data are returned as structures
%       with the pressure and particle velocity appended as separate
%       fields. In two dimensions, these fields are given by sensor_data.p,
%       sensor_data.ux, sensor_data.uy, and sensor_data.uz, and
%       field_final.p, field_final.ux, field_final.uy, and field_final.uz.
%
%       kspaceFirstOrder3D may also be used for time reversal image
%       reconstruction by assigning the time varying pressure recorded over
%       an arbitrary sensor surface to the input field
%       sensor.time_reversal_boundary_data. This data is then enforced in
%       time reversed order as a time varying Dirichlet boundary condition
%       over the sensor surface given by sensor.mask. The boundary data
%       must be indexed as
%       sensor.time_reversal_boundary_data(sensor_position, time). If
%       sensor.mask is given as a set of Cartesian coordinates, the
%       boundary data must be given in the same order. An equivalent binary
%       sensor mask (computed using nearest neighbour interpolation) is
%       then used to place the pressure values into the computational grid
%       at each time step. If sensor.mask is given as a binary grid of
%       sensor points, the boundary data must be ordered using MATLAB's
%       standard column-wise linear matrix indexing. If no additional
%       inputs are required, the source input can be replaced with an empty
%       array [].
%
%       Acoustic attenuation compensation can also be included during time
%       reversal image reconstruction by assigning the absorption
%       parameters medium.alpha_coeff and medium.alpha_power and reversing
%       the sign of the absorption term by setting medium.alpha_sign = [-1,
%       1]. This forces the propagating waves to grow according to the
%       absorption parameters instead of decay. The reconstruction should
%       then be regularised by assigning a filter to medium.alpha_filter
%       (this can be created using getAlphaFilter).
%
%       Note: To run a simple reconstruction example using time reversal
%       (that commits the 'inverse crime' of using the same numerical
%       parameters and model for data simulation and image reconstruction),
%       the sensor_data returned from a k-Wave simulation can be passed
%       directly to sensor.time_reversal_boundary_data  with the input
%       fields source.p0 and source.p removed or set to zero.
%
% USAGE:
%       sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor)
%       sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, ...) 
%
%       [sensor_data, field_data] = kspaceFirstOrder3D(kgrid, medium, source, sensor)
%       [sensor_data, field_data] = kspaceFirstOrder3D(kgrid, medium, source, sensor, ...) 
%
%       kspaceFirstOrder3D(kgrid, medium, source, [])
%       kspaceFirstOrder3D(kgrid, medium, source, [], ...) 
%
% INPUTS:
% The minimum fields that must be assigned to run an initial value problem
% (for example, a photoacoustic forward simulation) are marked with a *. 
%
%       kgrid*              - k-Wave grid structure returned by makeGrid
%                             containing Cartesian and k-space grid fields  
%       kgrid.t_array*      - evenly spaced array of time values [s] (set
%                             to 'auto' by makeGrid) 
%
%
%       medium.sound_speed* - sound speed distribution within the acoustic
%                             medium [m/s] 
%       medium.sound_speed_ref - reference sound speed used within the
%                             k-space operator (phase correction term)
%                             [m/s]
%       medium.density*     - density distribution within the acoustic
%                             medium [kg/m^3] 
%       medium.BonA         - parameter of nonlinearity
%       medium.alpha_power  - power law absorption exponent
%       medium.alpha_coeff  - power law absorption coefficient 
%                             [dB/(MHz^y cm)] 
%       medium.alpha_mode   - optional input to force either the absorption
%                             or dispersion terms in the equation of state
%                             to be excluded; valid inputs are
%                             'no_absorption' or 'no_dispersion' 
%       medium.alpha_filter - frequency domain filter applied to the
%                             absorption and dispersion terms in the
%                             equation of state 
%       medium.alpha_sign   - two element array used to control the sign of
%                             absorption and dispersion terms in the
%                             equation of state  
%
%
%       source.p0*          - initial pressure within the acoustic medium
%       source.p            - time varying pressure at each of the source
%                             positions given by source.p_mask 
%       source.p_mask       - binary grid specifying the positions of the
%                             time varying pressure source distribution
%       source.p_mode       - optional input to control whether the input
%                             pressure is injected as a mass source or
%                             enforced as a dirichlet boundary condition;
%                             valid inputs are 'additive' (the default) or
%                             'dirichlet'    
%       source.ux           - time varying particle velocity in the
%                             x-direction at each of the source positions
%                             given by source.u_mask 
%       source.uy           - time varying particle velocity in the
%                             y-direction at each of the source positions
%                             given by source.u_mask 
%       source.uz           - time varying particle velocity in the
%                             z-direction at each of the source positions
%                             given by source.u_mask  
%       source.u_mask       - binary grid specifying the positions of the
%                             time varying particle velocity distribution 
%       source.u_mode       - optional input to control whether the input
%                             velocity is applied as a force source or
%                             enforced as a dirichlet boundary condition;
%                             valid inputs are 'additive' (the default) or
%                             'dirichlet'
%
%
%       sensor.mask*        - binary grid or a set of Cartesian points
%                             where the pressure is recorded at each
%                             time-step  
%       sensor.time_reversal_boundary_data - time varying pressure
%                             enforced as a Dirichlet boundary condition
%                             over sensor.mask  
%       sensor.frequency_response - two element array specifying the center
%                             frequency and percentage bandwidth of a
%                             frequency domain Gaussian filter applied to
%                             the sensor_data
%       sensor.record_mode  - optional input to save the statistics of the
%                             wave field at the sensor elements rather than
%                             the time series; valid inputs are
%                             'statistics' or 'time_history' (the default)
%
% Note: For heterogeneous medium parameters, medium.sound_speed and
% medium.density must be given in matrix form with the same dimensions as
% kgrid. For homogeneous medium parameters, these can be given as single
% numeric values. If the medium is homogeneous and velocity inputs or
% outputs are not required, it is not necessary to specify medium.density.
%
% OPTIONAL INPUTS:
%       Optional 'string', value pairs that may be used to modify the
%       default computational settings.
%
%       'CartInterp'- Interpolation mode used to extract the pressure when
%                     a Cartesian sensor mask is given. If set to 'nearest'
%                     and more than one Cartesian point maps to the same
%                     grid point, duplicated data points are discarded and
%                     sensor_data will be returned with less points than
%                     that specified by sensor.mask (default = 'nearest').
%       'CreateLog' - Boolean controlling whether the command line output
%                     is saved using the diary function with a date and
%                     time stamped filename (default = false). 
%       'DataCast'  - String input of the data type that variables are cast
%                     to before computation. For example, setting to
%                     'single' will speed up the computation time (due to
%                     the improved efficiency of fftn and ifftn for this
%                     data type) at the expense of a loss in precision.
%                     This variable is also useful for utilising GPU
%                     parallelisation through libraries such as GPUmat or
%                     AccelerEyesJacket by setting 'DataCast' to
%                     'GPUsingle' or 'gsingle' (default = 'off').
%       'DisplayMask' - Binary matrix overlayed onto the animated
%                     simulation display. Elements set to 1 within the
%                     display mask are set to black within the display
%                     (default = sensor.mask).
%       'LogScale'  - Boolean controlling whether the pressure field is log
%                     compressed before display (default = false). The data
%                     is compressed by scaling both the positive and
%                     negative values between 0 and 1 (truncating the data
%                     to the given plot scale), adding a scalar value
%                     (compression factor) and then using the corresponding
%                     portion of a log10 plot for the compression (the
%                     negative parts are remapped to be negative thus the
%                     default color scale will appear unchanged). The
%                     amount of compression can be controlled by adjusting
%                     the compression factor which can be given in place of
%                     the Boolean input. The closer the compression factor
%                     is to zero, the steeper the corresponding part of the
%                     log10 plot used, and the greater the compression (the
%                     default compression factor is 0.02).
%       'MovieArgs' - Settings for movie2avi. Parameters must be given as
%                     {param, value, ...} pairs within a cell array
%                     (default = {}).
%       'MovieName' - Name of the movie produced when 'RecordMovie' is set
%                     to true (default = 'date-time-kspaceFirstOrder2D').
%       'PlotFreq'  - The number of iterations which must pass before the
%                     simulation plot is updated (default = 10).
%       'PlotLayout'- Boolean controlling whether three plots are produced
%                     of the initial simulation layout (initial pressure,
%                     sound speed, density) (default = false).
%       'PlotPML'   - Boolean controlling whether the perfectly matched
%                     layer is shown in the simulation plots. If set to
%                     false, the PML is not displayed (default = true).
%       'PlotScale' - [min, max] values used to control the scaling for
%                     imagesc (visualisation). If set to 'auto', a
%                     symmetric plot scale is chosen automatically for each
%                     plot frame.
%       'PlotSim'   - Boolean controlling whether the simulation iterations
%                     are progressively plotted (default = true).
%       'PMLAlpha'  - Absorption within the perfectly matched layer in
%                     Nepers per metre (default = 2).
%       'PMLInside' - Boolean controlling whether the perfectly matched
%                     layer is inside or outside the grid. If set to false,
%                     the input grids are enlarged by PMLSize before
%                     running the simulation (default = true). 
%       'PMLSize'   - Size of the perfectly matched layer in grid points.
%                     By default, the PML is added evenly to all sides of
%                     the grid, however, both PMLSize and PMLAlpha can be
%                     given as three element arrays to specify the x, y,
%                     and z properties, respectively. To remove the PML,
%                     set the appropriate PMLAlpha to zero rather than
%                     forcing the PML to be of zero size (default = 10).
%       'RecordMovie' - Boolean controlling whether the displayed image
%                     frames are captured and stored as a movie using
%                     movie2avi (default = false). 
%       'ReturnVelocity' - Boolean controlling whether the acoustic
%                     particle velocity at the positions defined by
%                     sensor.mask are also returned. If set to true, the
%                     output argument sensor_data is returned as a
%                     structure with the pressure and particle velocity
%                     appended as separate fields. In three dimensions,
%                     these fields are given by sensor_data.p,
%                     sensor_data.ux, sensor_data.uy, and sensor_data.uz.
%                     The field data is similarly returned as field_data.p,
%                     field_data.ux, field_data.uy, and field_data.uz.  
%       'Smooth'    - Boolean controlling whether source.p0,
%                     medium.sound_speed, and medium.density are smoothed
%                     using smooth before computation. 'Smooth' can either
%                     be given as a single Boolean value or as a 3 element
%                     array to control the smoothing of source.p0,
%                     medium.sound_speed, and medium.density,
%                     independently.  
%       'StreamToDisk' - Boolean controlling whether sensor_data is
%                     periodically saved to disk to avoid storing the
%                     complete matrix in memory. StreamToDisk may also be
%                     given as an integer which specifies the number of
%                     times steps that are taken before the data is saved
%                     to disk (default = 200).
%
% OUTPUTS:
% If 'ReturnVelocity' is set to false:
%       sensor_data - time varying pressure recorded at the sensor
%                     positions given by sensor.mask
%       field_data  - final pressure field
%
% If 'ReturnVelocity' is set to true:
%       sensor_data.p   - time varying pressure recorded at the sensor
%                         positions given by sensor.mask 
%       sensor_data.ux  - time varying particle velocity in the x-direction
%                         recorded at the sensor positions given by
%                         sensor.mask  
%       sensor_data.uy  - time varying particle velocity in the y-direction
%                         recorded at the sensor positions given by
%                         sensor.mask  
%       sensor_data.uz  - time varying particle velocity in the z-direction
%                         recorded at the sensor positions given by
%                         sensor.mask  
%       field_data.p    - final pressure field
%       field_data.ux   - final field of the particle velocity in the
%                         x-direction 
%       field_data.uy   - final field of the particle velocity in the
%                         y-direction 
%       field_data.uz   - final field of the particle velocity in the
%                         z-direction 
%
% ABOUT:
%       author      - Bradley Treeby and Ben Cox
%       date        - 7th April 2009
%       last update - 28th February 2012
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also fftn, ifftn, imagesc, kspaceFirstOrder1D, kspaceFirstOrder2D,
% makeGrid, makeTime, makeTransducer, smooth, unmaskSensorData 

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

% suppress mlint warnings that arise from using sub-scripts
%#ok<*NASGU>
%#ok<*COLND>
%#ok<*NODEF>
%#ok<*INUSL>

% start the timer and store the start time
start_time = clock;
tic;

% update command line status
disp('Running k-Wave simulation...');
disp(['  start time: ' datestr(start_time)]);

% =========================================================================
% DEFINE LITERALS
% =========================================================================

% minimum number of input variables
NUM_REQ_INPUT_VARIABLES = 4;

% optional input defaults
CARTESIAN_INTERP_DEF = 'nearest';
CREATE_LOG_DEF = false;
DATA_CAST_DEF = 'off';
DISPLAY_MASK_DEF = 'default';
LOG_SCALE_DEF = false;
LOG_SCALE_COMPRESSION_FACTOR_DEF = 0.02;
MOVIE_ARGS_DEF = {};
MOVIE_NAME_DEF = [getDateString '-kspaceFirstOrder3D'];
PLOT_FREQ_DEF = 10;
PLOT_LAYOUT_DEF = false;
PLOT_SCALE_DEF = [-1 1];
PLOT_SCALE_LOG_DEF = false;
PLOT_SIM_DEF = true;
PLOT_PML_DEF = true;
PML_ALPHA_DEF = 2;
PML_INSIDE_DEF = true;
PML_SIZE_DEF = 10;
RECORD_MOVIE_DEF = false;
RETURN_VELOCITY_DEF = false;
SAVE_TO_DISK_DEF = false;
SAVE_TO_DISK_FILENAME_DEF = 'kwave_input_data.mat';
SAVE_TO_DISK_EXIT_DEF = true;
SMOOTH_P0_DEF = true;
SMOOTH_C0_DEF = false;
SMOOTH_RHO0_DEF = false;
SOURCE_P_MODE_DEF = 'additive';
SOURCE_U_MODE_DEF = 'additive';
STREAM_TO_DISK_DEF = false;
STREAM_TO_DISK_STEPS_DEF = 200;
USE_KSPACE_DEF = true;
USE_SG_DEF = true;

% set default movie compression
MOVIE_COMP_WIN = 'Cinepak';
MOVIE_COMP_MAC = 'None';
MOVIE_COMP_LNX = 'None';
MOVIE_COMP_64B = 'None';

% set additional literals
MFILE = mfilename;
COLOR_MAP = getColorMap;
DT_WARNING_CFL = 0.45;  
ESTIMATE_SIM_TIME_STEPS = 50;
LOG_NAME = ['k-Wave-Log-' getDateString];
PLOT_SCALE_WARNING = 20;
STREAM_TO_DISK_FILENAME = 'temp_sensor_data.bin';

% =========================================================================
% CHECK INPUTS STRUCTURES AND OPTIONAL INPUTS
% =========================================================================

% run subscript to check inputs
kspaceFirstOrder_inputChecking;

% gpu memory counter for GPUmat toolbox
if strncmp(data_cast, 'kWaveGPU', 8);
    total_gpu_mem = GPUmem;
end

% =========================================================================
% UPDATE COMMAND LINE STATUS
% =========================================================================

disp(['  dt: ' scaleSI(dt) 's, t_end: ' scaleSI(t_array(end)) 's, time steps: ' num2str(length(t_array))]);
[x_sc scale prefix] = scaleSI(min([kgrid.x_size, kgrid.y_size, kgrid.z_size])); %#ok<ASGLU>
disp(['  input grid size: ' num2str(kgrid.Nx) ' by ' num2str(kgrid.Ny) ' by ' num2str(kgrid.Nz) ' grid points (' num2str(kgrid.x_size*scale) ' by ' num2str(kgrid.y_size*scale) ' by ' num2str(kgrid.z_size*scale) prefix 'm)']); 
if (kgrid.kx_max == kgrid.kz_max) && (kgrid.kx_max == kgrid.ky_max)
    disp(['  maximum supported frequency: ' scaleSI( kgrid.k_max * min(c(:)) / (2*pi) ) 'Hz']);
else
    disp(['  maximum supported frequency: ' scaleSI( kgrid.kx_max * min(c(:)) / (2*pi) ) 'Hz by ' scaleSI( kgrid.ky_max * min(c(:)) / (2*pi) ) 'Hz by ' scaleSI( kgrid.kz_max * min(c(:)) / (2*pi) ) 'Hz']);
end

% =========================================================================
% SMOOTH AND ENLARGE INPUT GRIDS
% =========================================================================

% smooth the initial pressure distribution p0 if required, and then restore
% the maximum magnitude (NOTE: if p0 has any values at the edge of the
% domain, the smoothing may cause part of p0 to wrap to the other side of
% the domain) 
if isfield(source, 'p0') && smooth_p0
    disp('  smoothing p0 distribution...');      
    source.p0 = smooth(kgrid, source.p0, true);
end

% expand the computational grid if the PML is set to be outside the input
% grid defined by the user (NOTE: the values of kgrid.t_array and kgrid.dt
% are not appended to the expanded grid)
if ~PML_inside

    % expand the computational grid, retaining the values for
    % kgrid.t_array
    disp('  expanding computational grid...');
    t_array_temp = kgrid.t_array;
    kgrid = makeGrid(kgrid.Nx + 2*PML_x_size, kgrid.dx, kgrid.Ny + 2*PML_y_size, kgrid.dy, kgrid.Nz + 2*PML_z_size, kgrid.dz);
    kgrid.t_array = t_array_temp;
    clear t_array_temp;
               
    % assign dt to kgrid if given as a structure
    if isstruct(kgrid)
        kgrid.dt = kgrid.t_array(2) - kgrid.t_array(1);
    end       
    
    % expand the grid matrices allowing a different PML size in each
    % Cartesian direction
    expand_size = [PML_x_size, PML_y_size, PML_z_size]; %#ok<NASGU>
    kspaceFirstOrder_expandGridMatrices;
    clear expand_size;
    
    % update command line status
    disp(['  computational grid size: ' num2str(kgrid.Nx) ' by ' num2str(kgrid.Ny) ' by ' num2str(kgrid.Nz) ' grid points']);

end

% define index values to remove the PML from the display if the optional
% input 'PlotPML' is set to false
if ~plot_PML
    % create indexes to allow inputs to be placed into the larger
    % simulation grid
    x1 = (PML_x_size + 1);
    x2 = kgrid.Nx - PML_x_size;
    y1 = (PML_y_size + 1);
    y2 = kgrid.Ny - PML_y_size;    
    z1 = (PML_z_size + 1);
    z2 = kgrid.Nz - PML_z_size;
else
    % create indexes to place the source input exactly into the simulation
    % grid
    x1 = 1;
    x2 = kgrid.Nx;
    y1 = 1;
    y2 = kgrid.Ny;
    z1 = 1;
    z2 = kgrid.Nz;
end    

% smooth the sound speed distribution if required
if smooth_c && numDim(c) == 3
    disp('  smoothing sound speed distribution...');      
    c = smooth(kgrid, c);
end
    
% smooth the ambient density distribution if required
if smooth_rho0 && numDim(rho0) == 3
    disp('  smoothing density distribution...');      
    rho0 = smooth(kgrid, rho0);
end

% =========================================================================
% PREPARE STAGGERED COMPUTATIONAL GRIDS AND OPERATORS
% =========================================================================

% interpolate the values of the density at the staggered grid locations
% where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z +
% dz/2)
if numDim(rho0) == 3 && use_sg
    
    % rho0 is heterogeneous and staggered grids are used
    rho0_sgx = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x + kgrid.dx/2, kgrid.y, kgrid.z, '*linear');
    rho0_sgy = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x, kgrid.y + kgrid.dy/2, kgrid.z, '*linear');
    rho0_sgz = interpn(kgrid.x, kgrid.y, kgrid.z, rho0, kgrid.x, kgrid.y, kgrid.z + kgrid.dz/2, '*linear');
    
    % set values outside of the interpolation range to original values
    rho0_sgx(isnan(rho0_sgx)) = rho0(isnan(rho0_sgx));
    rho0_sgy(isnan(rho0_sgy)) = rho0(isnan(rho0_sgy));    
    rho0_sgz(isnan(rho0_sgz)) = rho0(isnan(rho0_sgz));
    
else
    % rho0 is homogeneous or staggered grids are not used
    rho0_sgx = rho0;
    rho0_sgy = rho0;
    rho0_sgz = rho0;
end

% get the PML operators based on the reference sound speed and PML settings
pml_x = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, PML_x_size, PML_x_alpha, false, 1);
pml_x_sgx = getPML(kgrid.Nx, kgrid.dx, kgrid.dt, c_ref, PML_x_size, PML_x_alpha, true && use_sg, 1);
pml_y = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, PML_y_size, PML_y_alpha, false, 2);
pml_y_sgy = getPML(kgrid.Ny, kgrid.dy, kgrid.dt, c_ref, PML_y_size, PML_y_alpha, true && use_sg, 2);
pml_z = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, PML_z_size, PML_z_alpha, false, 3);
pml_z_sgz = getPML(kgrid.Nz, kgrid.dz, kgrid.dt, c_ref, PML_z_size, PML_z_alpha, true && use_sg, 3);

% define the k-space derivative operators, multiply by the staggered
% grid shift operators, and then re-order using ifftshift (the option
% use_sg exists for debugging) 
if use_sg
    ddx_k_shift_pos = ifftshift( 1i*kgrid.kx_vec .* exp(1i*kgrid.kx_vec*kgrid.dx/2) );
    ddx_k_shift_neg = ifftshift( 1i*kgrid.kx_vec .* exp(-1i*kgrid.kx_vec*kgrid.dx/2) );
    ddy_k_shift_pos = ifftshift( 1i*kgrid.ky_vec .* exp(1i*kgrid.ky_vec*kgrid.dy/2) );
    ddy_k_shift_neg = ifftshift( 1i*kgrid.ky_vec .* exp(-1i*kgrid.ky_vec*kgrid.dy/2) );
    ddz_k_shift_pos = ifftshift( 1i*kgrid.kz_vec .* exp(1i*kgrid.kz_vec*kgrid.dz/2) );
    ddz_k_shift_neg = ifftshift( 1i*kgrid.kz_vec .* exp(-1i*kgrid.kz_vec*kgrid.dz/2) );
else
    ddx_k_shift_pos = ifftshift( 1i*kgrid.kx_vec );
    ddx_k_shift_neg = ifftshift( 1i*kgrid.kx_vec );
    ddy_k_shift_pos = ifftshift( 1i*kgrid.ky_vec );
    ddy_k_shift_neg = ifftshift( 1i*kgrid.ky_vec );
    ddz_k_shift_pos = ifftshift( 1i*kgrid.kz_vec );
    ddz_k_shift_neg = ifftshift( 1i*kgrid.kz_vec );         
end

% create k-space operator (the option use_kspace exists for debugging)
if use_kspace
    kappa = ifftshift( sinc(c_ref*dt*kgrid.k/2) );
else
    kappa = 1;
end

% force the derivative operators and shift oeprators to be in the correct
% direction for use with BSXFUN
ddy_k_shift_pos = ddy_k_shift_pos.'; 
ddy_k_shift_neg = ddy_k_shift_neg.';
ddz_k_shift_pos = permute(ddz_k_shift_pos, [2 3 1]);
ddz_k_shift_neg = permute(ddz_k_shift_neg, [2 3 1]);

% cleanup unused variables
clear ax* ay* az* x0_min x0_max PML_x_alpha y0_min y0_max PML_y_alpha z0_min z0_max PML_z_alpha;

% =========================================================================
% PREPARE DATA MASKS AND STORAGE VARIABLES
% =========================================================================

% run subscript to create acoustic absorption variables
kspaceFirstOrder_createAbsorptionVariables;

% run subscript to create storage variables
kspaceFirstOrder_createStorageVariables;

% =========================================================================
% SCALE THE SOURCE TERMS
% =========================================================================

% run subscript to scale the source terms to the correct units
kspaceFirstOrder_scaleSourceTerms;

% =========================================================================
% SAVE DATA TO DISK FOR RUNNING SIMULATION EXTERNAL TO MATLAB
% =========================================================================

% save to disk option (currently in beta testing) for saving the input
% matrices to disk for running simulations outside of MATLAB
if save_to_disk
    % save files to disk
    kspaceFirstOrder_saveToDisk;
    
    % exit matlab computation if required
    if save_to_disk_exit
        return
    end
end

% =========================================================================
% DATA CASTING
% =========================================================================

% preallocate the loop variables
p = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
rhox = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
rhoy = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
rhoz = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
ux_sgx = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
uy_sgy = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
uz_sgz = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);
p_k = zeros(kgrid.Nx, kgrid.Ny, kgrid.Nz);

% run subscript to cast loop variables to other data types if an input is
% given for 'DataCast'
if ~strcmp(data_cast, 'off')
    kspaceFirstOrder_dataCast;
end

% =========================================================================
% CREATE INDEX VARIABLES
% =========================================================================

% setup the time index variable
if ~time_rev
    index_start = 1;
    index_step = 1;
    index_end = length(t_array); 
else
    % reverse the order of the input data
    sensor.time_reversal_boundary_data = fliplr(sensor.time_reversal_boundary_data);
    index_start = 1;
    index_step = 1;
    
    % stop one time point before the end so the last points are not
    % propagated
    index_end = length(t_array) - 1;
end

% =========================================================================
% PREPARE VISUALISATIONS
% =========================================================================

% pre-compute suitable axes scaling factor
if plot_layout || plot_sim
    [x_sc, scale, prefix] = scaleSI(max([kgrid.x_vec; kgrid.y_vec; kgrid.z_vec]));  %#ok<ASGLU>
end

% plot the simulation layout if 'PlotLayout' is set to true
if plot_layout
    kspaceFirstOrder_plotLayout;
end

% initialise the figures used for animation if 'PlotSim' is set to 'true'
if plot_sim
    img = figure;
    if ~time_rev
        pbar = waitbar(0, 'Computing Pressure Field', 'Visible', 'off');
    else
        pbar = waitbar(0, 'Computing Time Reversed Field', 'Visible', 'off');
    end
    
    % shift the waitbar so it doesn't overlap the figure window
    posn_pbar = get(pbar, 'OuterPosition');
    posn_img = get(img, 'OuterPosition');
    posn_pbar(2) = max(min(posn_pbar(2) - posn_pbar(4), posn_img(2) - posn_pbar(4) - 10), 0);
    set(pbar, 'OuterPosition', posn_pbar, 'Visible', 'on');
end 

% initialise movie parameters if 'RecordMovie' is set to 'true'
if record_movie
    kspaceFirstOrder_initialiseMovieParameters;
end

% =========================================================================
% LOOP THROUGH TIME STEPS
% =========================================================================

% update command line status
disp(['  precomputation completed in ' scaleTime(toc)]);
disp('  starting time loop...');

% restart timing variables
loop_start_time = clock;
tic;

% start time loop
for t_index = index_start:index_step:index_end

    % enforce time reversal bounday condition
    if time_rev

        % load pressure value and enforce as a Dirichlet boundary condition
        p(sensor_mask_ind) = sensor.time_reversal_boundary_data(:, t_index);

        % update p_k
        p_k = fftn(p);

        % compute rhox and rhoz using an adiabatic equation of state
        rhox_mod = p./(3*c.^2);
        rhoy_mod = p./(3*c.^2);
        rhoz_mod = p./(3*c.^2);
        rhox(sensor_mask_ind) = rhox_mod(sensor_mask_ind);
        rhoy(sensor_mask_ind) = rhoy_mod(sensor_mask_ind);
        rhoz(sensor_mask_ind) = rhoz_mod(sensor_mask_ind);
           
    end    

    % calculate ux, uy and uz at the next time step using dp/dx, dp/dy and
    % dp/dz at the current time step
    ux_sgx = bsxfun(@times, pml_x_sgx, ...
        bsxfun(@times, pml_x_sgx, ux_sgx) ... 
        - dt./rho0_sgx .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* p_k) )) ...
        );
    uy_sgy = bsxfun(@times, pml_y_sgy, ...
        bsxfun(@times, pml_y_sgy, uy_sgy) ...
        - dt./rho0_sgy .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* p_k) )) ...
        );
    uz_sgz = bsxfun(@times, pml_z_sgz, ...
        bsxfun(@times, pml_z_sgz, uz_sgz) ...
        - dt./rho0_sgz .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* p_k) )) ...
        );

    % force bsxfun compatability with Accelereyes GPU toolbox
    if strncmp(data_cast, 'g', 1);
        geval(ux_sgx, uy_sgy, uz_sgz);
    end
    
    % add in the velocity source terms   
    if ux_source >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            % enforce the source values as a dirichlet boundary condition
            ux_sgx(us_index) = source.ux(:, t_index);            
        else
            % add the source values to the existing field values
            ux_sgx(us_index) = ux_sgx(us_index) + source.ux(:, t_index);
        end
    end
    if uy_source >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            % enforce the source values as a dirichlet boundary condition
            uy_sgy(us_index) = source.uy(:, t_index);            
        else
            % add the source values to the existing field values
            uy_sgy(us_index) = uy_sgy(us_index) + source.uy(:, t_index);
        end        
    end
    if uz_source >= t_index
        if strcmp(source.u_mode, 'dirichlet')
            % enforce the source values as a dirichlet boundary condition
            uz_sgz(us_index) = source.uz(:, t_index);            
        else
            % add the source values to the existing field values
            uz_sgz(us_index) = uz_sgz(us_index) + source.uz(:, t_index);
        end         
    end   
    
    % add in transducer source term; there will normally be less source
    % points than time points, so this is only done until the source points
    % run out (the number of source points is stored in transducer_source)
    if transducer_source >= t_index
        % as only flat transducers are currently supported, assume all the
        % energy is transfered to x-direction velocity, multiply source
        % terms by apodization weights
        ux_sgx(us_index) = ux_sgx(us_index) + transducer_transmit_apodization.*transducer_input_signal(delay_mask);
        
        % update the delay_mask - this maps the source positions belonging
        % to different transducer elements to time points within
        % transducer_input_signal (this is a single time series) based on the
        % beamforming and focussing delays.  
        delay_mask = delay_mask + 1;
    end
    
    % calculate dux/dx, duydy and duz/dz at the next time step
    duxdx = real(ifftn( bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx)) ));
    duydy = real(ifftn( bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy)) ));
    duzdz = real(ifftn( bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz)) )); 

    % calculate rhox, rhoy and rhoz at the next time step
    if ~nonlinear
        % use linear mass conservation equation
        rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt.*rho0 .* duxdx);
        rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt.*rho0 .* duydy);        
        rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt.*rho0 .* duzdz);
    else
        % use nonlinear mass conversation equation
        rhox = bsxfun(@times, pml_x, ( bsxfun(@times, pml_x, rhox) - dt.*rho0 .* duxdx ) ./ (1 + 2*dt.*duxdx));
        rhoy = bsxfun(@times, pml_y, ( bsxfun(@times, pml_y, rhoy) - dt.*rho0 .* duydy ) ./ (1 + 2*dt.*duydy));
        rhoz = bsxfun(@times, pml_z, ( bsxfun(@times, pml_z, rhoz) - dt.*rho0 .* duzdz ) ./ (1 + 2*dt.*duzdz));
    end 
    
    % force bsxfun compatability with Accelereyes GPU toolbox
    if strncmp(data_cast, 'g', 1);
        geval(rhox, rhoy, rhoz);
    end
        
    % add in the pre-scaled pressure source term as a mass source   
    if p_source >= t_index
        if strcmp(source.p_mode, 'dirichlet')
            rhox(ps_index) = source.p(:, t_index);  
            rhoy(ps_index) = source.p(:, t_index);  
            rhoz(ps_index) = source.p(:, t_index);
        else
            rhox(ps_index) = rhox(ps_index) + source.p(:, t_index);
            rhoy(ps_index) = rhoy(ps_index) + source.p(:, t_index);  
            rhoz(ps_index) = rhoz(ps_index) + source.p(:, t_index);
        end
    end
    
    % calculate p at the next time step
    if ~nonlinear
        switch equation_of_state
            case 'lossless';
                % calculate p using a linear adiabatic equation of state
                p = c.^2.*(rhox + rhoy + rhoz);
            case 'absorbing';
                % calculate p using a linear absorbing equation of state                
                p = c.^2.*( ...
                    (rhox + rhoy + rhoz) ...
                    + absorb_tau.*real(ifftn( absorb_nabla1.*fftn(rho0.*(duxdx + duydy + duzdz)) )) ...
                    - absorb_eta.*real(ifftn( absorb_nabla2.*fftn(rhox + rhoy + rhoz) )) ...
                    );  
        end
    else
        switch equation_of_state
            case 'lossless';
                % calculate p using a nonlinear adiabatic equation of state
                p = c.^2.*(rhox + rhoy + rhoz + medium.BonA.*(rhox + rhoy + rhoz).^2./(2*rho0));
            case 'absorbing';
                % calculate p using a nonlinear absorbing equation of state 
                p = c.^2.*(...
                    (rhox + rhoy + rhoz) ...
                    + absorb_tau.*real(ifftn( absorb_nabla1.*fftn(rho0.*(duxdx + duydy + duzdz)) ))...
                    - absorb_eta.*real(ifftn( absorb_nabla2.*fftn(rhox + rhoy + rhoz) ))...
                    + medium.BonA.*(rhox + rhoy + rhoz).^2./(2*rho0) ...
                    );
        end
    end
             
    % enforce initial conditions if source.p0 is defined instead of time
    % varying sources
    if t_index == 1 && isfield(source, 'p0')
    
        % add the initial pressure to rho as a mass source
        p = source.p0;
        rhox = source.p0./(3*c.^2);
        rhoy = source.p0./(3*c.^2);
        rhoz = source.p0./(3*c.^2);

        % compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)
        % which forces u(t = t1) = 0
        ux_sgx = dt./rho0_sgx .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) / 2;
        uy_sgy = dt./rho0_sgy .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) / 2;
        uz_sgz = dt./rho0_sgz .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) / 2;    
        
    end
    
    % precompute fft of p here so p can be modified for visualisation
    p_k = fftn(p);        
  
    % update index for data storage - if streaming to disk, a smaller
    % matrix is used which is continually overwritten, and then saved to
    % disk each time it is filled
    if stream_to_disk
        file_index = t_index - stream_to_disk*(stream_data_index - 1);
    else
        file_index = t_index;
    end
    
    % extract required sensor data from the pressure and particle
    % velocity fields
    switch extract_data_case
        case 1
            % return velocity = false
            % binary sensor mask = false
            F_interp.V = reshape(p, [], 1);
            sensor_data(:, file_index) = F_interp(sensor_x, sensor_y, sensor_z);
            
        case 2
            % return velocity = false
            % binary sensor mask = true  
            if transducer_sensor
                if transducer_receive_elevation_focus
                    
                    % update the sensor data buffer
                    sensor_data_buffer = circshift(sensor_data_buffer, [0 1]);
                    sensor_data_buffer(:, 1) = p(sensor_mask_ind);
                    
                    % if buffer has been filled, store average pressure
                    % across each transducer element accounting for the
                    % elevation focus
                    if t_index >= sensor_data_buffer_size
                        % get the current values
                        current_vals = sum(transducer_receive_mask.*sensor_data_buffer, 2);
                        
                        % reshape and average
                        sensor_data(:, file_index - sensor_data_buffer_size + 1) = ...
                            sum(reshape(sum(reshape(current_vals, [], sensor.element_length), 2), sensor.element_width, []).', 2);
                    end
                    
                else
                    % store average pressure across each transducer element
                    sensor_data(:, file_index) = sum(reshape(sum(reshape(p(sensor_mask_ind), [], sensor.element_length), 2), sensor.element_width, []).', 2);
                end
            else            
                if store_time_series   
                    % store the function values for each time point
                    sensor_data(:, file_index) = p(sensor_mask_ind);
                else
                    % store only the cumulative statistics
                    sensor_data.p_rms(:) = sqrt((sensor_data.p_rms(:).^2*(t_index - 1) + p(sensor_mask_ind).^2)./t_index);
                    sensor_data.p_max(:) = max(sensor_data.p_max(:), p(sensor_mask_ind));                
                end
            end
            
        case 3
            % return velocity = true
            % binary sensor mask = false           
            F_interp.V = reshape(p, [], 1);
            sensor_data.p(:, file_index) = F_interp(sensor_x, sensor_y, sensor_z);
            F_interp.V = reshape(ux_sgx, [], 1);
            sensor_data.ux(:, file_index) = F_interp(sensor_x, sensor_y, sensor_z);
            F_interp.V = reshape(uy_sgy, [], 1);
            sensor_data.uy(:, file_index) = F_interp(sensor_x, sensor_y, sensor_z);            
            F_interp.V = reshape(uz_sgz, [], 1);
            sensor_data.uz(:, file_index) = F_interp(sensor_x, sensor_y, sensor_z);     
            
        case 4
            % return velocity = true
            % binary sensor mask = true
            sensor_data.p(:, file_index) = p(sensor_mask_ind);
            sensor_data.ux(:, file_index) = ux_sgx(sensor_mask_ind);
            sensor_data.uy(:, file_index) = uy_sgy(sensor_mask_ind);            
            sensor_data.uz(:, file_index) = uz_sgz(sensor_mask_ind);     
            
    end        

    % if the data is being streamed to disk and sensor_data has just been
    % filled, append the values of sensor_data to the values already saved
    % to disk
    if stream_to_disk && (t_index == stream_to_disk*stream_data_index)

        % open the file to append the new data
        if stream_data_index == 1
            % create or open the file and overwrite any existing data
            try
                fid = fopen(STREAM_TO_DISK_FILENAME, 'w+');
            catch ME
                disp('Error in writing file using ''StreamToDisk''');
                rethrow(ME);
            end
        else
            % open the file and append new data
            fid = fopen(STREAM_TO_DISK_FILENAME, 'a+');
        end
        
        % write values at end of file using the precision specified in
        % data_cast
        if strcmp(data_cast, 'off')
            fwrite(fid, sensor_data, 'double');
        elseif strcmp(data_cast, 'single')
            fwrite(fid, sensor_data, 'single');            
        elseif strcmp(data_cast, 'gsingle') || strcmp(data_cast, 'kWaveGPUsingle')
            fwrite(fid, single(sensor_data), 'single');
        elseif strcmp(data_cast, 'gdouble') || strcmp(data_cast, 'kWaveGPUdouble')
            fwrite(fid, double(sensor_data), 'double');
        else
            error('Unknown ''DataCast'' options used with ''StreamToDisk''');            
        end
            
        % close the file
        fclose(fid);
 
        % increment the data file index if there is
        % still data left
        if index_end > t_index
            stream_data_index = stream_data_index + 1;
        end
        
    end

    % estimate the time to run the simulation
    if t_index == ESTIMATE_SIM_TIME_STEPS
        disp(['  estimated simulation time ' scaleTime(etime(clock, loop_start_time)*index_end/t_index) '...']);

        % display current matlab memory usage
        if nargout == 3 && strncmp(computer, 'PCWIN', 5)
            [mem_usage.user, mem_usage.sys] = memory;
            disp(['  memory used: ' num2str(mem_usage.user.MemUsedMATLAB./1024^3) ' GB (of ' num2str(mem_usage.sys.PhysicalMemory.Total./1024^3) ' GB)']); 
        end
                
        % gpu memory counter for GPUmat toolbox
        if strncmp(data_cast, 'kWaveGPU', 8)
            current_gpu_mem = GPUmem;
            disp(['  GPU memory used: ' num2str((total_gpu_mem - current_gpu_mem)/2^30) ' GB (of ' num2str(total_gpu_mem/2^30) ' GB)']);
            mem_usage.gpu.total = total_gpu_mem;
            mem_usage.gpu.used = total_gpu_mem - current_gpu_mem;
        end
        
        % gpu memory counter for Accelereyes toolbox
        if strncmp(data_cast, 'g', 1)
            gpu_info = ginfo(true);
            disp(['  GPU memory used: ' num2str((gpu_info.gpu_total - gpu_info.gpu_free)/2^30) ' GB (of ' num2str(gpu_info.gpu_total/2^30) ' GB)']);
            mem_usage.gpu.total = gpu_info.gpu_total;
            mem_usage.gpu.used = gpu_info.gpu_total - gpu_info.gpu_free;            
        end
    end
    
    % plot data if required
    if plot_sim && (rem(t_index, plot_freq) == 0 || t_index == 1 || t_index == index_end) 
        
        % update progress bar
        waitbar(t_index/length(t_array), pbar);
        drawnow;

        % ensure p is cast as a CPU variable
        p_plot = double(p(x1:x2, y1:y2, z1:z2));       

        % update plot scale if set to automatic or log
        if plot_scale_auto || plot_scale_log
            kspaceFirstOrder_adjustPlotScale;
        end         
        
        % add display mask onto plot
        if strcmp(display_mask, 'default')
            p_plot(sensor.mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(2);
        elseif ~strcmp(display_mask, 'off')
            p_plot(display_mask(x1:x2, y1:y2, z1:z2) ~= 0) = plot_scale(2);
        end     

        % update plot
        planeplot(scale*kgrid.x_vec(x1:x2), scale*kgrid.y_vec(y1:y2), scale*kgrid.z_vec(z1:z2), p_plot, '', plot_scale, prefix, COLOR_MAP);

        % save movie frames if required
        if record_movie
            
            % set background color to white
            set(gcf, 'Color', [1 1 1]);

            % save the movie frame
            movie_frames(frame_index) = getframe(gcf);

            % update frame index
            frame_index  = frame_index  + 1;
            
        end
        
        % update variable used for timing variable to exclude the first
        % time step if plotting is enabled
        if t_index == 1
            loop_start_time = clock;
        end        
    end
end

% assign the final time reversal values
if time_rev
    p(sensor_mask_ind) = sensor.time_reversal_boundary_data(:, index_end + 1);
end

% update command line status
disp(['  simulation completed in ' scaleTime(toc)]);

% =========================================================================
% CLEAN UP
% =========================================================================

% save final sensor_data variable to disk if required
if stream_to_disk

    % clear some time loop variables before reloading sensor data to
    % free up memory (in case the sensor data is very large)
    clear duxdx duydy duzdz ux_sgx uy_sgy uz_sgz rhox rhoy rhoz;
    
    % double check the data has not just been saved
    if t_index ~= stream_to_disk*stream_data_index
        
        % open the file to append the new data
        if stream_data_index == 1
            % open the file and overwrite any existing data
            fid = fopen(STREAM_TO_DISK_FILENAME, 'w+');
        else
            % open the file and append new data
            fid = fopen(STREAM_TO_DISK_FILENAME, 'a+');
        end
        
        % extract required data
        sensor_data = sensor_data(:, 1:t_index - stream_to_disk*(stream_data_index - 1));
        
        % write values at end of file
        if strcmp(data_cast, 'off')
            fwrite(fid, sensor_data, 'double');
        elseif strcmp(data_cast, 'single')
            fwrite(fid, sensor_data, 'single');            
        elseif strcmp(data_cast, 'gsingle') || strcmp(data_cast, 'kWaveGPUsingle')
            fwrite(fid, single(sensor_data), 'single');
        elseif strcmp(data_cast, 'gdouble') || strcmp(data_cast, 'kWaveGPUdouble')
            fwrite(fid, double(sensor_data), 'double');
        end
            
        % close the file
        fclose(fid);
        
    end
       
    % reload complete streamed data and assign to sensor_data
    fid = fopen(STREAM_TO_DISK_FILENAME, 'r');
    if strcmp(data_cast, 'off') || strcmp(data_cast, 'gdouble') || strcmp(data_cast, 'kWaveGPUdouble')
        sensor_data = fread(fid, [sum(sensor.mask(:)), length(t_array)], 'double');
    elseif strcmp(data_cast, 'single') || strcmp(data_cast, 'gsingle') || strcmp(data_cast, 'kWaveGPUsingle')
        sensor_data = fread(fid, [sum(sensor.mask(:)), length(t_array)], 'single');            
    else
        error('Unknown ''DataCast'' options used with ''StreamToDisk''');
    end
    fclose(fid);    
    
    % permanently delete the temporary storage
    disp('  removing temporary data...');
    old_state = recycle;
    recycle('off');
    delete(STREAM_TO_DISK_FILENAME);
    recycle(old_state);
   
end

% save the movie frames to disk
if record_movie
    kspaceFirstOrder_saveMovieFile;   
end

% clean up used figures
if plot_sim
    close(img);
    close(pbar);
end

% reset the indexing variables to allow original grid size to be maintained
% (this is used to remove the PML from the user data if 'PMLInside' is set
% to false)
if (~plot_PML && PML_inside)
    x1 = x1 - PML_x_size;
    x2 = x2 + PML_x_size;
    y1 = y1 - PML_y_size;
    y2 = y2 + PML_y_size;    
    z1 = z1 - PML_z_size;
    z2 = z2 + PML_z_size;
elseif (plot_PML && ~PML_inside)
    x1 = x1 + PML_x_size;
    x2 = x2 - PML_x_size;
    y1 = y1 + PML_y_size;
    y2 = y2 - PML_y_size;
    z1 = z1 + PML_z_size;
    z2 = z2 - PML_z_size;    
end

% save the final pressure field if in time reversal mode
if time_rev
    sensor_data = p(x1:x2, y1:y2, z1:z2);
end

% save the final pressure and velocity fields if required
if return_velocity
    
    % cast variables back to double
    if ~time_rev && use_sensor
        sensor_data.p = double(sensor_data.p);
        sensor_data.ux = double(sensor_data.ux);
        sensor_data.uy = double(sensor_data.uy);
        sensor_data.uz = double(sensor_data.uz);      
    end
    
    % return final field data if required by user
    if nargout >= 2
        field_data.p = double(p(x1:x2, y1:y2, z1:z2));
        field_data.ux = double(ux_sgx(x1:x2, y1:y2, z1:z2));
        field_data.uy = double(uy_sgy(x1:x2, y1:y2, z1:z2));
        field_data.uz = double(uz_sgz(x1:x2, y1:y2, z1:z2));
    end
        
else

    % cast variables back to double
    if use_sensor
        if store_time_series
            sensor_data = double(sensor_data);
        else
            sensor_data.p_rms = double(sensor_data.p_rms);
            sensor_data.p_max = double(sensor_data.p_max);            
        end
    end
    
    % return final field data if required by user
    if nargout >= 2
        field_data = double(p(x1:x2, y1:y2, z1:z2));        
    end
        
end

% process the sensor data if recorded using a transducer
if transducer_sensor
    % the pressure values across the grid points in each element have
    % already been automatically summed, so now scale the summed pressure
    % data by the number of grid points in each element to convert to an
    % average 
    sensor_data = sensor_data./(sensor.element_length * sensor.element_width);
end
    
% reorder the sensor points if a binary sensor mask was used for Cartesian
% sensor mask nearest neighbour interpolation
if use_sensor && reorder_data
    
    % update command line status
    disp('  reordering Cartesian measurement data...');
    
    if return_velocity
        
        % append the reordering data
        new_col_pos = length(sensor_data.p(1,:)) + 1;
        sensor_data.p(:, new_col_pos) = reorder_index;
        sensor_data.ux(:, new_col_pos) = reorder_index;
        sensor_data.uy(:, new_col_pos) = reorder_index;
        sensor_data.uz(:, new_col_pos) = reorder_index;        

        % reorder p0 based on the order_index
        sensor_data.p = sortrows(sensor_data.p, new_col_pos);
        sensor_data.ux = sortrows(sensor_data.ux, new_col_pos);
        sensor_data.uy = sortrows(sensor_data.uy, new_col_pos);
        sensor_data.uz = sortrows(sensor_data.uz, new_col_pos);

        % remove the reordering data
        sensor_data.p = sensor_data.p(:, 1:new_col_pos - 1);
        sensor_data.ux = sensor_data.ux(:, 1:new_col_pos - 1);
        sensor_data.uy = sensor_data.uy(:, 1:new_col_pos - 1);
        sensor_data.uz = sensor_data.uz(:, 1:new_col_pos - 1);   
        
    else
        
        % append the reordering data
        new_col_pos = length(sensor_data(1,:)) + 1;
        sensor_data(:, new_col_pos) = reorder_index;

        % reorder p0 based on the order_index
        sensor_data = sortrows(sensor_data, new_col_pos);

        % remove the reordering data
        sensor_data = sensor_data(:, 1:new_col_pos - 1);
        
    end
end

% filter the recorded time domain pressure signals if transducer filter
% parameters are given 
if use_sensor && ~time_rev && isfield(sensor, 'frequency_response')
    sensor_data = gaussianFilter(sensor_data, 1/kgrid.dt, sensor.frequency_response(1), sensor.frequency_response(2));
end

% resize the transducer object if the grid has been expanded
if ~PML_inside

    % set the size to trim from the grid
    retract_size = [PML_x_size, PML_y_size, PML_z_size]; 
    
    % check if the sensor is a transducer
    if strcmp(class(sensor), 'kWaveTransducer')
        
        % retract the transducer mask
        sensor.retract_grid(retract_size);
        
    end
        
    % check if the source is a transducer, and if so, and different
    % transducer to the sensor 
    if strcmp(class(source), 'kWaveTransducer') && ~(strcmp(class(sensor), 'kWaveTransducer') && isequal(sensor, source))
        
        % retract the transducer mask
        source.retract_grid(retract_size);
        
    end
end

% return empty sensor data if not used
if ~use_sensor
    sensor_data = [];
end

% update command line status
disp(['  total computation time ' scaleTime(etime(clock, start_time))]);

% switch off log
if create_log
    diary off;
end

function planeplot(x_vec, y_vec, z_vec, data, data_title, plot_scale, prefix, color_map)
% Subfunction to produce a plot of a three-dimensional matrix through the
% three central planes
   
subplot(2, 2, 1), imagesc(y_vec, x_vec, squeeze(data(:, :, round(end/2))), plot_scale);
title([data_title 'x-y plane']);
axis image;

subplot(2, 2, 2), imagesc(z_vec, x_vec, squeeze(data(:, round(end/2), :)), plot_scale);
title('x-z plane');
axis image;
xlabel(['(All axes in ' prefix 'm)']);

subplot(2, 2, 3), imagesc(z_vec, y_vec, squeeze(data(round(end/2), :, :)), plot_scale);
title('y-z plane');
axis image;
colormap(color_map); 
drawnow;