function varargout = spectrum(func, Fs, varargin)
%SPECTRUM   Compute the single sided amplitude and phase spectrums.
%
% DESCRIPTION:
%       spectrum computes the single sided amplitude and phase spectrums
%       of an arbitrary one-dimensional input signal.
%
% USAGE:
%       func_as = spectrum(func, Fs)
%       func_as = spectrum(func, Fs, ...)
%       [f, func_as] = spectrum(func, Fs)
%       [f, func_as] = spectrum(func, Fs, ...)
%       [f, func_as, func_ps] = spectrum(func, Fs)
%       [f, func_as, func_ps] = spectrum(func, Fs, ...)
%
% INPUTS:
%       func        - signal to analyse
%       Fs          - sampling frequency [Hz]
%
% OPTIONAL INPUTS:
%       Optional 'string', value pairs that may be used to modify the
%       default computational settings.
%
%       'FFTLength' - Length of the FFT. If the set length is larger than
%                     the signal length, the signal is zero padded. If the
%                     set length is smaller than the signal length, the
%                     default value is used instead (default = signal
%                     length).  
%       'PowerTwo'  - Boolean controlling whether the FFT length is forced
%                     to be the next highest power of 2 (default = false)
%       'Plot'      - Boolean controlling whether the amplitude and phase
%                     spectrums are plotted (default = false). Can be given
%                     as a two element array to control the plot of the
%                     amplitude and phase spectrums, respectively.
%       'Unwrap'    - Boolean controlling whether the phase spectrum is
%                     unwrapped (default = false)
%       'Window'    - parameter string controlling the window type used to
%                     filter the signal before the FFT is taken (default =
%                     'Rectangular'). Any valid input types for getWin may
%                     be used.
%
% OUTPUTS:
%       f           - frequency array
%       func_as     - single sided amplitude spectrum
%       func_ps     - single sided phase spectrum
%
% ABOUT:
%       author      - Bradley E. Treeby
%       date        - 12th June 2009
%       last update - 27th July 2010
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2012 Bradley Treeby and Ben Cox
%
% See also fft, unwrap

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

% set usage defaults
num_req_input_variables = 2;
window = 'Rectangular';
power_two = false;
plot_fft = false;
unwrap_phase = false;

% replace with user defined values if provided
if nargin < num_req_input_variables
    error('Incorrect number of inputs');
elseif ~isempty(varargin)
    for input_index = 1:2:length(varargin)
        switch varargin{input_index}
            case 'FFTLength'
                fft_length = varargin{input_index + 1};
            case 'Window'
                window = varargin{input_index + 1};
            case 'PowerTwo'             
                power_two = varargin{input_index + 1}(1);
            case 'Plot'
                plot_fft = varargin{input_index + 1};    
            case 'Unwrap'
                unwrap_phase = varargin{input_index + 1};                  
            otherwise
                error('Unknown optional input');
        end
    end
end

% force the input to be a column vector
sz = size(func);
if length(sz) ~= 2 || (sz(1) ~= 1 && sz(2) ~= 1)
    error('Input function must be a vector');
elseif sz(1) > sz (2)
    func = func.';
end

% assign the number of points
func_length = max(sz);

% set the length of the FFT
if ~(exist('fft_length', 'var') && fft_length > func_length)
    if power_two
        % find an appropriate FFT length of the form 2^N that is equal to or
        % larger than the length of the input signal
        fft_length = 2^(nextpow2(func_length));
    else
        % set the FFT length to the function length
        fft_length = func_length;
    end
end

% window the signal
[win, coherent_gain] = getWin(func_length, window, 'Symmetric', false);
func = func.*win.';

% compute the fft using the defined FFT length, if fft_length >
% func_length, the input signal is padded with zeros
func_fft = fft(func, fft_length);

% correct for the magnitude scaling of the FFT and the coherent gain of the
% window (note that the correction is equal to func_length NOT fft_length)
func_fft = func_fft/(func_length*coherent_gain);

% reduce to a single sided spectrum where the number of unique points for
% even numbered FFT lengths is given by N/2 + 1, and for odd (N + 1)/2
num_unique_pts = ceil((fft_length+1)/2);
func_fft = func_fft(1:num_unique_pts);

% correct the single-sided magnitude by multiplying the symmetric points by
% 2 (the DC and Nyquist components are unique and are not multiplied by 2
% and the Nyquist component only exists for even numbered FFT lengths)
if rem(fft_length, 2)
    % odd FFT length
    func_fft(2:end) = func_fft(2:end)*2;        
else
    % even FFT length
    func_fft(2:end-1) = func_fft(2:end-1)*2;    
end

% create the frequency axis variable
f = (0:length(func_fft)-1)*Fs/fft_length;

% calculate the amplitude spectrum
func_as = abs(func_fft);

% calculate the phase spectrum
func_ps = angle(func_fft);

% unwrap the phase spectrum if required
if unwrap_phase
    func_ps = unwrap(func_ps);
end

% plot the amplitude spectrum of the input signal if required
if plot_fft(1)
    figure;
    [f_sc scale prefix] = scaleSI(max(f)); %#ok<ASGLU>
    plot(f*scale, func_as, 'k-');
    xlabel(['Frequency [' prefix 'Hz]']);
    ylabel('Signal Amplitude');
end

% plot the phase spectrum of the input signal if required
if plot_fft(end)
    figure;
    [f_sc scale prefix] = scaleSI(max(f)); %#ok<ASGLU>
    plot(f*scale, func_ps, 'k-');
    xlabel(['Frequency [' prefix 'Hz]']);
    ylabel('Signal Phase');
end

% assign the outputs
if nargout == 1
    varargout(1) = {func_as};
elseif nargout == 2;
    varargout(1) = {f};
    varargout(2) = {func_as};
elseif nargout == 3;
    varargout(1) = {f};
    varargout(2) = {func_as};
    varargout(3) = {func_ps};    
end