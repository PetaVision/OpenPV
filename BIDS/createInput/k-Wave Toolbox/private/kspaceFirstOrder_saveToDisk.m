% DESCRIPTION:
%       subscript to save input data to disk
%
% ABOUT:
%       author      - Bradley Treeby and Jiri Jaros
%       date        - 24th August 2011
%       last update - 24th February 2012
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

% update command line status
disp(['  precomputation completed in ' scaleTime(toc)]);
tic;
disp('  saving input files to disk...');

% =========================================================================
% VARIABLE LIST
% =========================================================================

% list of all the static variables used within the time loop
variable_list = {'dt', ...
    'pml_x_sgx', 'pml_y_sgy', 'pml_z_sgz', 'pml_x', 'pml_y', 'pml_z', ...
    'rho0', 'rho0_sgx', 'rho0_sgy', 'rho0_sgz', 'c', ...    
    'kappa_r', ...
    'ddx_k_shift_pos_r', 'ddx_k_shift_neg_r', ...
    'ddy_k_shift_pos', 'ddy_k_shift_neg', ... 
    'ddz_k_shift_pos', 'ddz_k_shift_neg', ...        
    'absorb_tau', 'absorb_eta', ...
    'absorb_nabla1_r', 'absorb_nabla2_r', ...    
    'BonA', ...
    };

% list of all the integer variables used within the time loop regardless of
% options
integer_variable_list = {'sensor_mask_ind', 'sensor_mask_index_size',...
    'Nx', 'Ny', 'Nz', 'Nx_r', 'Nt',...
    'ux_source_flag', 'uy_source_flag', 'uz_source_flag', ...
    'p_source_flag', 'p0_source_flag', 'transducer_source_flag',...
    };

% =========================================================================
% PREPARE AND CREATE REQUIRED C/C++ VARIABLES
% =========================================================================

% assign pseudo-names for the source flags
ux_source_flag = ux_source;
uy_source_flag = uy_source;
uz_source_flag = uz_source;
p_source_flag = p_source;
p0_source_flag = isfield(source, 'p0');
transducer_source_flag = transducer_source;

% source modes and indicies
% - these are only defined if the source flags are > 0
% - the source mode describes whether the source will be added or replaced
% - the source indicies describe which grid points act as the source
% - the us_index is reused for any of the u sources and the transducer source
if ux_source_flag || uy_source_flag || uz_source_flag
    u_source_mode = ~strcmp(source.u_mode, 'dirichlet');
    if ux_source_flag
        u_source_many = numDim(source.ux) > 1;
    elseif uy_source_flag
        u_source_many = numDim(source.uy) > 1;
    elseif uz_source_flag
        u_source_many = numDim(source.uz) > 1;
    end
    us_index_size = squeeze(length(us_index));
    integer_variable_list = [integer_variable_list, {'u_source_mode', 'u_source_many', 'us_index', 'us_index_size'}];
end
if p_source_flag
    p_source_mode = ~strcmp(source.p_mode, 'dirichlet');
    ps_index_size = squeeze(length(ps_index));
    p_source_many = numDim(source.p) > 1;
    integer_variable_list = [integer_variable_list, {'p_source_mode', 'p_source_many', 'ps_index', 'ps_index_size'}];
end
if transducer_source_flag
    us_index_size = squeeze(length(us_index));
    integer_variable_list = [integer_variable_list, {'us_index', 'us_index_size'}];
end

% source variables
% - these are only defined if the source flags are > 0
% - these are the actual source values
% - these are indexed as (position_index, time_index)
if ux_source_flag
    ux_source_input = source.ux;
    variable_list = [variable_list, {'ux_source_input'}];
end
if uy_source_flag
    uy_source_input = source.uy;
    variable_list = [variable_list, {'uy_source_input'}];
end
if uz_source_flag 
    uz_source_input = source.uz;
    variable_list = [variable_list, {'uz_source_input'}];
end
if p_source_flag
    p_source_input = source.p;
    variable_list = [variable_list, {'p_source_input'}];
end
if transducer_source_flag
    transducer_source_input = transducer_input_signal;
    variable_list = [variable_list, {'transducer_source_input'}];
    integer_variable_list = [integer_variable_list, {'delay_mask'}];
end

% initial pressure source variable
% - this is only defined if the p0 source flag is 1
% - this defines the initial pressure everywhere (there is no indicies)
if p0_source_flag
    p0_source_input = source.p0;
    variable_list = [variable_list, {'p0_source_input'}];
end

% assign remaining pseudo-names for the variables stored in structures
Nx = kgrid.Nx;
Ny = kgrid.Ny;
Nz = kgrid.Nz;
Nt = length(t_array);

% make sure the medium parameters are given as matrices
if numel(c) == 1
    c = c*ones(Nx, Ny, Nz);
end

if numel(medium.BonA) == 1
    BonA = medium.BonA*ones(Nx, Ny, Nz);
else
    BonA = medium.BonA;
end

if numel(rho0) == 1
    rho0 = rho0*ones(Nx, Ny, Nz);
    rho0_sgx = rho0;
    rho0_sgy = rho0;
    rho0_sgz = rho0;
end

if numel(absorb_tau) == 1
    absorb_tau = absorb_tau*ones(Nx, Ny, Nz);
end

if numel(absorb_eta) == 1
    absorb_eta = absorb_eta*ones(Nx, Ny, Nz);
end

% create reduced variables for use with real-to-complex FFT (saves memory)
Nx_r = Nx/2 + 1;
kappa_r = kappa(1:Nx_r, :, :);
ddx_k_shift_pos_r = ddx_k_shift_pos(1:Nx_r);
ddx_k_shift_neg_r = ddx_k_shift_neg(1:Nx_r);
absorb_nabla1_r = absorb_nabla1(1:Nx_r, :, :); 
absorb_nabla2_r = absorb_nabla2(1:Nx_r, :, :);

% create scalar size variables
sensor_mask_index_size = squeeze(length(sensor_mask_ind));

% =========================================================================
% DATACAST AND SAVING
% =========================================================================

% change all the variables to be in single precision
data_cast = 'single';
for cast_index = 1:length(variable_list)
    eval([variable_list{cast_index} ' = ' data_cast '(' variable_list{cast_index} ');']);
end

% change all the index variables to be in unsigned integers
data_cast = 'uint64';
for cast_index = 1:length(integer_variable_list)
    eval([integer_variable_list{cast_index} ' = ' data_cast '(' integer_variable_list{cast_index} ');']);
end
 
% save the input variables to disk as a MATLAB binary file
variable_list = [variable_list, integer_variable_list];
save(save_to_disk, variable_list{:});   

% separate the filename parts
[save_to_disk_pathstr, save_to_disk_name, save_to_disk_ext] = fileparts(save_to_disk);

% save the integer variables to disk in a seperate file
save([save_to_disk_pathstr filesep save_to_disk_name '_vars' save_to_disk_ext], integer_variable_list{:});   

% update command line status
disp(['  completed in ' scaleTime(toc)]);