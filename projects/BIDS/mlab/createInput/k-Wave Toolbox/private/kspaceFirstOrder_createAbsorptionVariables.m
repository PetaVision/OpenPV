% DESCRIPTION:
%       subscript to create the absorption variables
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 26th November 2010
%       last update - 7th February 2012
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

% define the lossy derivative operators and proportionality coefficients
if strcmp(equation_of_state, 'absorbing')
            
    % make sure the operators are positive and real
    medium.alpha_coeff = abs(real(medium.alpha_coeff));
    medium.alpha_power = abs(real(medium.alpha_power));
    
    % convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
    medium.alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power);

    % compute the absorbing fractional Laplacian operator and coefficient
    if ~(isfield(medium, 'alpha_mode') && strcmp(medium.alpha_mode, 'no_absorption'))
        absorb_nabla1 = (kgrid.k).^(medium.alpha_power-2); 
        absorb_nabla1(isinf(absorb_nabla1)) = 0;
        absorb_nabla1 = ifftshift(absorb_nabla1);
        absorb_tau = -2*medium.alpha_coeff.*c.^(medium.alpha_power - 1);
    else
        absorb_nabla1 = 0;
        absorb_tau = 0;
    end
       
    % compute the dispersive fractional Laplacian operator and coefficient
    if ~(isfield(medium, 'alpha_mode') && strcmp(medium.alpha_mode, 'no_dispersion'))
        absorb_nabla2 = (kgrid.k).^(medium.alpha_power-1); 
        absorb_nabla2(isinf(absorb_nabla2)) = 0;
        absorb_nabla2 = ifftshift(absorb_nabla2);            
        absorb_eta = 2*medium.alpha_coeff.*c.^(medium.alpha_power)*tan(pi*medium.alpha_power/2);
    else
        absorb_nabla2 = 0;
        absorb_eta = 0;
    end
        
    % pre-filter the absorption parameters if alpha_filter is defined (this
    % is used for time-reversal photoacoustic image reconstruction
    % with absorption compensation)
    if isfield(medium, 'alpha_filter');
                
        % update command line status
        disp('  filtering absorption variables...');        
        
        % frequency shift the absorption parameters
        absorb_nabla1 = fftshift(absorb_nabla1);
        absorb_nabla2 = fftshift(absorb_nabla2);
                        
        % apply the filter
        absorb_nabla1 = absorb_nabla1.*medium.alpha_filter;
        absorb_nabla2 = absorb_nabla2.*medium.alpha_filter;

        % shift the parameters back
        absorb_nabla1 = ifftshift(absorb_nabla1);
        absorb_nabla2 = ifftshift(absorb_nabla2); 
           
    end    

    % modify the sign of the absorption operators if alpha_sign is defined
    % (this is used for time-reversal photoacoustic image reconstruction
    % with absorption compensation)
    if isfield(medium, 'alpha_sign')
       if numel(medium.alpha_sign) == 2
           % if two parameters are given, apply the first to the absorption
           % parameter and the second to the disperion parameters
           absorb_tau = sign(medium.alpha_sign(1))*absorb_tau;
           absorb_eta = sign(medium.alpha_sign(2))*absorb_eta;
       else
           error('medium.alpha_sign must be given as a 2 element array controlling absorption and dispersion, respectively.');
       end
    end
end