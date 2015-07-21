function att = attenuationWater(f, T)
%ATTENUATIONWATER   Calculate ultrasound attenuation in distilled water
%
% DESCRIPTION:
%       attenuationWater calculates the ultrasonic absorption in distilled
%       water at a given temperature and frequency using a 7th order
%       polynomial fitted to the data given by Pinkerton (1949). 
%
% USAGE:
%       att = attenuationWater(f, T)
%
% INPUTS:
%       f   - array of frequency values [MHz]
%       T   - water temperature [degC]
%
% OUTPUTS:
%       att - attenuation [dB/cm]
%
% ABOUT:
%       author      - Bradley E. Treeby
%       date        - 10th November 2008
%       last udpate - 10th November 2008 
%
% REFERENCES:
%   [1] Pinkerton (1949) "The Absorption of Ultrasonic Waves in Liquids and
%       its Relation to Molecular Constitution," Proceedings of the
%       Physical Society. Section B, 2, 129-141
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also speedSoundWater

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

if T < 0 || T > 60
    disp('WARNING: Temperature outside range of experimental data');
end

% conversion factor between Nepers and dB
NEPER2DB = 8.686;       

% coefficients for 7th order polynomial fit
a_0 = 56.723531840522710;
a_1 = -2.899633796917384;
a_2 = 0.099253401567561;
a_3 = -0.002067402501557;
a_4 = 2.189417428917596e-005;
a_5 = -6.210860973978427e-008;
a_6 = -6.402634551821596e-010;
a_7 = 3.869387679459408e-012;

% compute attenuation
a_on_fsqr = (a_0 + a_1*T + a_2*T.^2 + a_3*T.^3 + a_4*T.^4 + a_5*T.^5 + a_6*T.^6 + a_7*T.^7)*1e-17;
att = NEPER2DB*1e12*f.^2*a_on_fsqr;