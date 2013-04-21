% DESCRIPTION:
%       subscript modify p_plot if optional input 'PlotScale' is set to
%       'auto' or 'decibel'
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 4th February 2011
%       last update - 10th February 2011
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

% update plot scale if set to automatic
if plot_scale_auto
    mx = max(abs( p_plot(:)) );
    if mx == 0
        plot_scale = [-0.1, 0.1];
    else
        plot_scale = [-mx, mx];
    end
end

% rescale plot variable if plot scale set to log
if plot_scale_log
    
    % update plot scale if set to auto
    if plot_scale_auto
        alt_plot_scale_lin = plot_scale;
        alt_plot_scale_log = log10(abs(plot_scale) + log_scale_comp_factor) - log10(log_scale_comp_factor);
        alt_plot_scale_log(1) = -alt_plot_scale_log(1);
    else
        % truncate data to the given plot scale
        p_plot(p_plot < alt_plot_scale_lin(1)) = alt_plot_scale_lin(1);
        p_plot(p_plot > alt_plot_scale_lin(2)) = alt_plot_scale_lin(2);
    end

    % scale both positive and negative data separately
    p_plot(p_plot > 0) = log10(p_plot(p_plot > 0) + log_scale_comp_factor) - log10(log_scale_comp_factor);
    p_plot(p_plot < 0) = -(log10(-p_plot(p_plot < 0) + log_scale_comp_factor) - log10(log_scale_comp_factor));
    
    % update plot scale
    plot_scale = alt_plot_scale_log;
end