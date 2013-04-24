function mat_rs = resize(varargin)
%RESIZE     Resize a matrix.
%
% DESCRIPTION:
%       Resize a matrix to a given size using interp2 (2D) or interp3 (3D).
%
% USAGE:
%       mat_rs = resize(mat, new_size)
%       mat_rs = resize(mat, new_size, interp_mode)
%
% INPUTS:
%       mat         - matrix to resize
%       new_size    - desired matrix size in elements given by [Nx, Ny] in
%                     2D and [Nx, Ny, Nz] in 3D. Here Nx is the number of
%                     elements in the row direction, Ny is the number of
%                     elements in the column direction, and Nz is the
%                     number of elements in the depth direction.
%
% OPTIONAL INPUTS:
%       interp_mode - interpolation mode used by interp2 and interp3 
%                     (default = '*linear')
%
% OUTPUTS:
%       mat_rs      - resized matrix
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 8th July 2009
%       last update - 26th October 2011
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also interp2, interp3

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

% check the inputs for release B.0.2 compatability
if length(varargin{2}) == 1 && nargin >= 3 && length(varargin{3}) == 1
    
    % display warning message
    disp('WARNING: input usage deprecated, please see documentation.');
    disp('In future releases this usage will no longer be functional.');    
    
    % recursively call resize with the correct inputs
    if nargin == 3
        mat_rs = resize(varargin{1}, [varargin{2}, varargin{3}]);
    else
        mat_rs = resize(varargin{1}, [varargin{2}, varargin{3}], varargin{4});
    end
    return
    
end

% start the timer
tic;

% update command line status
disp('Resizing matrix...');

% assign the matrix input
mat = varargin{1};

% check for interpolation mode input
if nargin == 2
    interp_mode = '*linear';
elseif nargin ~= 3
    error('incorrect number of inputs');
else
    interp_mode = varargin{3};
end

% check inputs
if numDim(mat) ~= length(varargin{2})
    error('resolution input must have the same number of elements as data dimensions');
end

switch numDim(mat)
    case 2
        % extract the original number of pixels from the size of the matrix
        [Nx_input, Ny_input] = size(mat);
        
        % extract the desired number of pixels
        Nx_output = varargin{2}(1);
        Ny_output = varargin{2}(2);

        % update command line status
        disp(['  input grid size: ' num2str(Nx_input) ' by ' num2str(Ny_input) ' elements']);
        disp(['  output grid size: ' num2str(Nx_output) ' by ' num2str(Ny_output) ' elements']);         
        
        % check the size is different to the input size
        if Nx_input ~= Nx_output || Ny_input ~= Ny_output 

            % resize the input matrix to the desired number of pixels
            mat_rs = interp2(0:1/(Ny_input - 1):1, (0:1/(Nx_input - 1):1)', mat, 0:1/(Ny_output - 1):1, (0:1/(Nx_output - 1):1)', interp_mode);

        else
            mat_rs = mat;
        end
    case 3
        
        % extract the original number of pixels from the size of the matrix
        [Nx_input, Ny_input, Nz_input] = size(mat);

        % extract the desired number of pixels
        Nx_output = varargin{2}(1);
        Ny_output = varargin{2}(2); 
        Nz_output = varargin{2}(3);        
        
        % update command line status
        disp(['  input grid size: ' num2str(Nx_input) ' by ' num2str(Ny_input) ' by ' num2str(Nz_input) ' elements']);
        disp(['  output grid size: ' num2str(Nx_output) ' by ' num2str(Ny_output) ' by ' num2str(Nz_output) ' elements']); 

        % create normalised plaid grids of current discretisation
        [x_mat, y_mat, z_mat] = ndgrid((0:Nx_input-1)/(Nx_input-1), (0:Ny_input-1)/(Ny_input-1), (0:Nz_input-1)/(Nz_input-1));       

        % create plaid grids of desired discretisation
        [x_mat_interp, y_mat_interp, z_mat_interp] = ndgrid((0:Nx_output-1)/(Nx_output-1), (0:Ny_output-1)/(Ny_output-1), (0:Nz_output-1)/(Nz_output-1));

        % compute interpolation; for a matrix indexed as [M, N, P], the
        % axis variables must be given in the order N, M, P
        mat_rs = interp3(y_mat, x_mat, z_mat, mat, y_mat_interp, x_mat_interp, z_mat_interp, interp_mode);        
        
    otherwise
        error('input matrix must be 2 or 3 dimensional');
end

% update command line status
disp(['  completed in ' scaleTime(toc)]);    