%KWAVEGRID   Class definition for k-space grid structure.
%
% DESCRIPTION:
%       See makeGrid for function arguments.
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 22nd July 2010
%       last update - 19th July 2011
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

classdef kWaveGrid    
    
    % define the properties (these parameters are stored)
    properties
        % grid dimensions in pixels/voxels
        Nx = 0;
        Ny = 0;
        Nz = 0;
        
        % pixel/voxel size in metres
      	dx = 0;
        dy = 0;
        dz = 0;
        
        % wavenumber grids
        kx_vec = 0;
        ky_vec = 0;
        kz_vec = 0;           
        k = 0;
        
        % maximum supported spatial frequency
        kx_max = 0;
        ky_max = 0;
        kz_max = 0;
        k_max = 0;
        
        % time array
        t_array = 'auto';
        
        % number of dimensions
        dim = 0;
    end
    
    % define the dependent properties (these parameters are computed each
    % time they are needed)
    properties(Dependent = true)
        % Cartesian spatial grids
        x;
        y;
        z;
        
        % 3D plaid wavenumber grids
        kx;
        ky;
        kz;
        
        % 1D spatial grids
        x_vec;
        y_vec;
        z_vec;    
        
        % Cartesian grid size
        x_size;
        z_size;
        y_size;
        
        % time step and number of time points
        dt;
        Nt;
        
        % total number of grid points
        total_grid_points;
    end
    
    % constructor function
    methods
        
        function kgrid = kWaveGrid(varargin)
            
            % assign the input values to the grid object
            if nargin == 6
                kgrid.Ny = varargin{3};
                kgrid.dy = varargin{4};
                kgrid.Nz = varargin{5};
                kgrid.dz = varargin{6};
            elseif nargin == 4
                kgrid.Ny = varargin{3};
                kgrid.dy = varargin{4};
            elseif nargin ~= 2
                error('Incorrect number of input arguments');
            end
            kgrid.Nx = varargin{1};
            kgrid.dx = varargin{2};
                       
            switch nargin
                case 2
                    % assign the grid parameters for the x spatial direction
                    kgrid.kx_vec = kgrid.makeDim(kgrid.Nx, kgrid.dx);
                   
                    % define the scalar wavenumber based on the wavenumber components
                    kgrid.k = abs(kgrid.kx_vec);
                    
                    % define maximum supported frequency
                    kgrid.kx_max = max(abs(kgrid.kx_vec(:)));
                    kgrid.k_max = kgrid.kx_max;
                    
                    % set the number of dimensions
                    kgrid.dim = 1;
                case 4
                    % assign the grid parameters for the x and z spatial directions
                    kgrid.kx_vec = kgrid.makeDim(kgrid.Nx, kgrid.dx);
                    kgrid.ky_vec = kgrid.makeDim(kgrid.Ny, kgrid.dy);

                    % define plaid grids of the wavenumber components centered about 0
                    [kx, ky] = ndgrid(kgrid.kx_vec, kgrid.ky_vec);

                    % define the scalar wavenumber based on the wavenumber components
                    kgrid.k = sqrt(kx.^2 + ky.^2);               
                    
                    % define maximum supported frequency
                    kgrid.kx_max = max(abs(kgrid.kx_vec(:)));
                    kgrid.ky_max = max(abs(kgrid.ky_vec(:)));
                    kgrid.k_max = min([kgrid.kx_max, kgrid.ky_max]);    
                    
                    % set the number of dimensions
                    kgrid.dim = 2;                    
                case 6
                    % assign the grid parameters for the x ,y and z spatial directions
                    kgrid.kx_vec = kgrid.makeDim(kgrid.Nx, kgrid.dx);
                    kgrid.ky_vec = kgrid.makeDim(kgrid.Ny, kgrid.dy);
                    kgrid.kz_vec = kgrid.makeDim(kgrid.Nz, kgrid.dz);

                    % define plaid grids of the wavenumber components centered about 0
                    [kx, ky, kz] = ndgrid(kgrid.kx_vec, kgrid.ky_vec, kgrid.kz_vec);

                    % define the scalar wavenumber based on the wavenumber components
                    kgrid.k = sqrt(kx.^2 + ky.^2 + kz.^2);
                   
                    % define maximum supported frequency
                    kgrid.kx_max = max(abs(kgrid.kx_vec(:)));
                    kgrid.ky_max = max(abs(kgrid.ky_vec(:)));
                    kgrid.kz_max = max(abs(kgrid.kz_vec(:)));
                    kgrid.k_max = min([kgrid.kx_max, kgrid.ky_max, kgrid.kz_max]);  
                    
                    % set the number of dimensions
                    kgrid.dim = 3;                    
            end
        end
    end
    
    % functions for dependent variables that only run when queried
    methods
        
        function x = get.x(obj)
            % calculate x based on kx
            x = obj.x_size*obj.kx*obj.dx/(2*pi);
        end
        
        function y = get.y(obj)
            % calculate y based on ky
            y = obj.y_size*obj.ky*obj.dy/(2*pi);
        end
        
        function z = get.z(obj)
            % calculate z based on kz
            z = obj.z_size*obj.kz*obj.dz/(2*pi);
        end        
        
        function kx = get.kx(obj)
            % duplicate kx vector
            switch obj.dim
                case 1
                    kx = obj.kx_vec;
                case 2
                    kx = repmat(obj.kx_vec, [1, obj.Ny]);
                case 3
                    kx = repmat(obj.kx_vec, [1, obj.Ny, obj.Nz]);
            end
        end
        
        function ky = get.ky(obj)
            % rotate ky vector to y direction then duplicate
            switch obj.dim
                case 1
                    ky = 0;
                case 2
                    ky = repmat(obj.ky_vec.', [obj.Nx, 1]);
                case 3
                    ky = repmat(obj.ky_vec.', [obj.Nx, 1, obj.Nz]);
            end
        end     
        
        function kz = get.kz(obj)
            % permute kz vector to z direction then duplicate
            switch obj.dim
                case 1
                    kz = 0;
                case 2
                    kz = 0;
                case 3
                    kz = repmat(permute(obj.kz_vec, [2 3 1]), [obj.Nx, obj.Ny, 1]);
            end
        end
  
        function x_vec = get.x_vec(obj)
            % calculate x_vec based on kx_vec
            x_vec = obj.x_size*obj.kx_vec*obj.dx/(2*pi);
        end
        
        function y_vec = get.y_vec(obj)
            % calculate y_vec based on ky_vec
            y_vec = obj.y_size*obj.ky_vec*obj.dy/(2*pi);
        end
        
        function z_vec = get.z_vec(obj)
            % calculate z_vec based on kz_vec
            z_vec = obj.z_size*obj.kz_vec*obj.dz/(2*pi);
        end        
                        
        function x_size = get.x_size(obj)
            % calculate x_size based on Nx and dx
            x_size = obj.Nx*obj.dx;
        end
        
        function y_size = get.y_size(obj)
            % calculate y_size based on Ny and dy
            y_size = obj.Ny*obj.dy;
        end
        
        function z_size = get.z_size(obj)
            % calculate z_size based on Nz and dz
            z_size = obj.Nz*obj.dz;
        end     
        
        function dt = get.dt(obj)
            % extract dt from the time array
            if strcmp(obj.t_array, 'auto')
                dt = 'auto';
            else
                dt = obj.t_array(2) - obj.t_array(1);
            end
        end           
        
        function Nt = get.Nt(obj)
            % extract dt from the time array
            if strcmp(obj.t_array, 'auto')
                Nt = 'auto';
            else
                Nt = length(obj.t_array);
            end
        end          
        
        function N = get.total_grid_points(obj)
            % calculate the total number of points in the grid
            switch obj.dim
                case 1
                    N = obj.Nx;
                case 2
                    N = obj.Nx*obj.Ny;
                case 3
                    N = obj.Nx*obj.Ny*obj.Nz;
            end
        end
    end
   
    % functions that can only be accessed by class members
    methods (Access = 'protected', Static = true) 
        
        % subfunction to create the grid parameters for a single spatial direction
        function kx_vec = makeDim(Nx, dx)

            % define the discretisation of the spatial dimension such that there is
            % always a DC component
            if rem(Nx, 2) == 0
                % grid dimension has an even number of points
                nx = (-0.5:1/Nx:0.5-1/Nx).';
            else
                % grid dimension has an odd number of points
                nx = (-0.5:1/(Nx-1):0.5).';
            end

            % force middle value to be zero in case 1/Nx is a recurring
            % number and the series doesn't give exactly zero
            nx(floor(Nx/2) + 1) = 0;
            
            % define the wavenumber vector components
            kx_vec = (2*pi/dx).*nx;       
        end
        
    end
end

