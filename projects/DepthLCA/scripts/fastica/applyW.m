function [Out1, Out2] = applyW(mixedsig, W, ...)

# File:  applyW.m 
#
# Description: Script for applying the inverse mixing matrix W to a matrix of data
#
# Examples: 
# [icasig, meanvalues] = applyW(mixedsig, W)
# Gives the ica matrix and the mean values of mixedsig.  These means are not put 
# back into icasig.
#
# [icasig] = applyW(mixedsig, W, 'mean')
# Gives the ica matrix only, with the means put back into icasig.
#
# Created: February, 2002, Dan Ryan

if nargin==2
  [mixedsig, meanvalues] = remmean(mixedsig);
  icasig = W * mixedsig;
elseif (nargin<2)
  error('Not enough arguments');
  exit(-1);
elseif (nargin>3)
  error('Too many arguments');
  exit(-1);
elseif nargin==3
  param = va_arg();
  if strcmp('mean', param)
    icasig = W * mixedsig;
  else
    error(['Unrecognized parameter: ''' param '''']);
    exit(-1);
  endif
endif

# Determine what to output
if nargout == 1
  Out1 = icasig;
else
  Out1 = icasig;
  Out2 = meanvalues;
endif
# Plot to make sure they look reasonable:
icaplot('histogram', icasig);

# Say how big the ascii files are:
length = size(icasig, 2);
printf("The ascii files will have %i data points.\n", length);

endfunction





