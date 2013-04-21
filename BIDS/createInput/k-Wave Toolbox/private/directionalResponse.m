function p_sensor = directionalResponse(kgrid, sensor, p_k)
%DIRECTIONALRESPONSE    Apply directivity to the sensor measurements.
% 
% DESCRIPTION:
%       directionalResponse takes a pressure field in k-space, p_k, 
%       multiplies by a directivity function, and outputs the pixels
%       indicated in sensor.mask.
%
% USAGE:
%       p_sensor = directionalResponse(kgrid, sensor, p_k)
%
% INPUTS:
%       kgrid       - k-space grid structure returned by makeGrid
%       sensor.directivity_angle   - a matrix with the same structure as
%                     sensor.mask that allocates a directivity angle to
%                     each sensor element as defined in sensor.mask. The
%                     angles are in radians:
%                     0 = max sensitivity in y direction (up/down)
%                     pi/2 or -pi/2 = max sensitivity in x direction (left/right)
%       sensor.directivity_pattern - a text string with currently only one
%                     option. 'pressure' indicates that the directional
%                     response should be of the kind due to spatial
%                     averaging over a sensor surface, so a sinc function
%                     in 2D.
%       sensor.directivity_size    - the directivity pattern used is what
%                     would be the directivity if the sensor were this
%                     length (width). The larger this is the more
%                     directional the response.
%       p_k         - the acoustic pressure field in the k-space domain.
%
% 
% Currently works for binary sensor_mask, but not when sensor_mask is given
% as Cartesian coordinates. Also, currently works only in 2D.
% 
% ABOUT:
%       author: Ben Cox
%       date: 21st January 2010
%       last update: 13th December 2011
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

% DEFAULTS
DIRECTIVITY_PATTERN_DEF = 'pressure';
DIRECTIVITY_SIZE_DEF = max([kgrid.dx,kgrid.dy]);

% unload sensor structure and apply defaults
if ~isfield(sensor,'directivity_pattern')
    sensor.directivity_pattern = DIRECTIVITY_PATTERN_DEF;
end
if ~isfield(sensor,'directivity_size')
    sensor.directivity_size = DIRECTIVITY_SIZE_DEF;
end

% check sensor.directivity_pattern and sensor.mask have the same size
if sum(size(sensor.directivity_angle) ~= size(sensor.mask))
    error('sensor.directivity_angle and sensor.mask must have the same structure')
end

% sensor mask indices
sensor_mask_ind  = find(sensor.mask ~= 0);
Ns = length(sensor_mask_ind);

% find the unique directivity angles
unique_angles = unique(sensor.directivity_angle(sensor_mask_ind));

% calculate the number of unique angles (which is number of loops required)
Nunique = length(unique_angles);

% pre-allocate p_sensor vector
p_sensor = zeros(Ns,1);

for loop = 1:Nunique
   
   theta = unique_angles(loop);

   % find which of the sensors have this directivity
   indices = find(sensor.directivity_angle(sensor_mask_ind) == theta);

   switch sensor.directivity_pattern
      case 'pressure'
         % calculate magnitude of component of wavenumber along sensor face
         k_tangent = reshape([cos(theta) -sin(theta)]*[kgrid.ky(:)'; kgrid.kx(:)'], kgrid.Nx, kgrid.Ny);
         directionality = fftshift(sinc(k_tangent*sensor.directivity_size/2));         
      case 'gradient'
         % calculate magnitude of component of wavenumber normal to the sensor face          
         k_normal = reshape([sin(theta), cos(theta)]*[kgrid.ky(:)'; kgrid.kx(:)'], kgrid.Nx, kgrid.Ny);
         temp = k_normal./kgrid.k;
         temp(kgrid.k==0) = 0;
         directionality = fftshift(temp);         
      otherwise
         error('Unsupported directivity pattern')
   end
   
   % apply the directivity response to the pressure field (in k-space)
   p_directivity = real(ifft2(p_k.*directionality));
   
   % pick out the response at the sensor points
   p_sensor(indices) = p_directivity(sensor_mask_ind(indices));
   
end

