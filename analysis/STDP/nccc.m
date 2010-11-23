function [Q, deltaT] = nccc(st1, st2, tau, maxT, T, verbose)
% [Q, tr] = nccc(st1, st2, tau, maxT, T, verbose)
% Normalized cipogram with 2nd order statistics.
%
% Input (seconds)
%   st1, st2: spike trains with sorted spike timings
%   tau: time constant for CIP kernel
%   maxT: correlogram range will be effective in [-maxT, maxT]
%   T: length of spike train in seconds
%   verbose: (optional/0) 
%   
% Output
%   Q: normalized continuous-time correlogram
%   deltaT: time range
% 
% Published in:
% Il Park, Antonio R. C. Paiva, Jose Principe, Thomas B. DeMarse.
% An Efficient Algorithm for Continuous-time Cross Correlogram of Spike Trains,
% Journal of Neuroscience Methods, Volume 168, Issue 2, 15 March 2008, 514-523
% doi:10.1016/j.jneumeth.2007.10.005
%
% Copyright 2007 Antonio and Memming, CNEL, all rights reserved
% Contact: memming <at> gmail <dot> com
% $Id: nccc.m 78 2009-11-10 20:50:05Z memming $

[Q, deltaT] = ccc(st1, st2, tau, maxT, T, verbose);

N1 = length(st1);
N2 = length(st2);
Nij = N1 * N2;

Q = (Q * T - Nij / T) * 2 * sqrt(tau * T) / sqrt(Nij);
