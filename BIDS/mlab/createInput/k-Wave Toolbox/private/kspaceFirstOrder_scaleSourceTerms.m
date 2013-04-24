% DESCRIPTION:
%       subscript to scale source terms to the correct units
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 15th February 2012
%       last update - 15th February 2012
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

% get the dimension size
N = kgrid.dim;

% scale the input pressure by 1/c^2 (to convert to units of density), by
% 1/N (to split the input across the split density field), and if the
% pressure is injected as a mass source, also scale the pressure by
% 2*dt*c/dx to account for the time step and convert to units of 
% [kg/(m^3 s)]
if p_source 
    if strcmp(source.p_mode, 'dirichlet')  
        if numel(c) == 1
            % compute the scale parameter based on the homogeneous sound speed
            source.p = source.p ./ (N*c^2);
        else
            % compute the scale parameter seperately for each source position
            % based on the sound speed at that position
            for p_index = 1:length(source.p(:,1))        
                source.p(p_index, :) = source.p(p_index, :) ./ c(ps_index(p_index))^2;
            end
        end
    else
        if numel(c) == 1
            % compute the scale parameter based on the homogeneous sound speed
            source.p = source.p .* (2*dt./(N*c*kgrid.dx));
        else
            % compute the scale parameter seperately for each source position
            % based on the sound speed at that position
            for p_index = 1:length(source.p(:,1))        
                source.p(p_index, :) = source.p(p_index, :) .* (2.*dt./(N*c(ps_index(p_index)).*kgrid.dx));
            end
        end
    end
end

% if source.u_mode is not set to 'dirichlet', scale the x-direction
% velocity source terms by 2*dt*c/dx to account for the time step and
% convert to units of [m/s^2] 
if ux_source && ~strcmp(source.u_mode, 'dirichlet')
    if numel(c) == 1
        % compute the scale parameter based on the homogeneous sound speed
        source.ux = source.ux .* (2*c*dt./kgrid.dx);
    else
        % compute the scale parameter seperately for each source position
        % based on the sound speed at that position
        for u_index = 1:length(source.ux(:,1))
            source.ux(u_index, :) = source.ux(u_index, :) .* (2.*c(us_index(u_index)).*dt./kgrid.dx);
        end
    end
end

% if source.u_mode is not set to 'dirichlet', scale the y-direction
% velocity source terms by 2*dt*c/dy to account for the time step and
% convert to units of [m/s^2] 
if uy_source && ~strcmp(source.u_mode, 'dirichlet')
    if numel(c) == 1
        % compute the scale parameter based on the homogeneous sound speed
        source.uy = source.uy .* (2*c*dt./kgrid.dy);
    else
        % compute the scale parameter seperately for each source position
        % based on the sound speed at that position
        for u_index = 1:length(source.uy(:,1))
            source.uy(u_index, :) = source.uy(u_index, :) .* (2.*c(us_index(u_index)).*dt./kgrid.dy);
        end
    end
end 

% if source.u_mode is not set to 'dirichlet', scale the z-direction
% velocity source terms by 2*dt*c/dz to account for the time step and
% convert to units of [m/s^2]  
if uz_source && ~strcmp(source.u_mode, 'dirichlet')     
    if numel(c) == 1
        source.uz = source.uz .* (2*c*dt./kgrid.dz);
    else
        % compute the scale parameter seperately for each source position
        % based on the sound speed at that position
        for u_index = 1:length(source.uz(:,1))        
            source.uz(u_index, :) = source.uz(u_index, :) .* (2.*c(us_index(u_index)).*dt./kgrid.dz);
        end
    end
end

% scale the transducer source term by 2*dt*c/dx to account for the time
% step and convert to units of [m/s^2] 
if transducer_source   
    if numel(c) == 1
        transducer_input_signal = transducer_input_signal .* (2*c*dt./kgrid.dx);
    else
        % compute the scale parameter based on the average sound speed at the
        % transducer positions (only one input signal is used to drive the
        % transducer)
        transducer_input_signal = transducer_input_signal.* (2*(mean(c(us_index)))*dt./kgrid.dx);
    end
end

% clear subscript variables
clear N