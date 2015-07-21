function [Q, deltaT] = ccc(st1, st2, tau, maxT, T, verbose)
% [Q, tr] = ccc(st1, st2, tau, maxT, T, verbose)
%
% Input (seconds)
%   st1, st2: spike trains with sorted spike timings
%   tau: time constant for CIP kernel
%   maxT: correlogram range will be effective in [-maxT, maxT]
%   T: length of spike train in seconds
%   verbose: (optional/0) detailed info, uses tic, toc
%   
% Output
%   Q: continuous-time correlogram
%   deltaT: time range
%
% See also: nccc
%
% Published in:
% Il Park, Antonio R. C. Paiva, Jose Principe, Thomas B. DeMarse.
% An Efficient Algorithm for Continuous-time Cross Correlogram of Spike Trains,
% Journal of Neuroscience Methods, Volume 168, Issue 2, 15 March 2008, 514-523
% doi:10.1016/j.jneumeth.2007.10.005
%
% Copyright 2007 Antonio and Memming, CNEL, all rights reserved
% $Id: ccc.m 80 2010-01-22 23:10:10Z memming $

if nargin < 5
    verbose = 0;
end

N1 = length(st1);
N2 = length(st2);
Nij = N1 * N2;

if N1 == 0 || N2 == 0
    warning('ccc:NODATA', 'At least one spike is required!');
    deltaT = []; Q = [];
    return;
end

maxTTT = abs(maxT) + tau * 10; % exp(-100) is effectively zero

% rough estimate of # of time difference required (assuming independence)
% this estimate is not good if the spike trains are strongly correlated
eN = ceil((max(N1, N2))^2 * maxTTT * 2 / min(st1(end), st2(end)));
if verbose; fprintf('Expected time differences [%d] / [%d]\n', eN, Nij); end
deltaT = zeros(2 * eN, 1); 

% Compute all the time differences
if verbose; fprintf('Extracting time differences...\n'); tic; end
lastStartIdx = 1;
k = 1;
for n = 1:N1
    incIdx = find((st2(lastStartIdx:N2) - st1(n) >= -maxTTT), 1, 'first');
    lastStartIdx = lastStartIdx + incIdx - 1;
    % disp(lastStartIdx);
    for m = lastStartIdx:N2
	timeDiff = st2(m) - st1(n);
	if timeDiff <= maxTTT
	    deltaT(k) = timeDiff;
	    k = k + 1;
	else % this is the ending point
	    n = N1 + 1;
	    break;
	end
    end
end
if verbose; fprintf('Extracting time differences finished [%f sec]\r', toc); end

deltaT = deltaT(1:(k-1));
N = length(deltaT);
if N < 2
    warning('ccc:NODATA', 'At least two intervals are required');
    deltaT = []; Q = [];
    return;
end
if verbose
    fprintf('Actual number of time differences [%d]\nSorting...\n', N); tic;
end

deltaT = sort(deltaT, 1); % Sort the time differences
if verbose; fprintf('Sorting finished [%f sec]\r', toc); end

Qplus = zeros(N, 1);
Qminus = zeros(N, 1);
Qminus(1) = 1;
Qplus(N) = 0;

EXP_DELTA = exp(-(diff(deltaT))/tau);
for k = 1:(N-1)
    Qminus(k + 1) = 1 + Qminus(k) * EXP_DELTA(k);
    kk = N - k;
    Qplus(kk) =  (Qplus(kk+1) + 1) * EXP_DELTA(kk);
end

Q = Qminus + Qplus;
