## Copyright (C) 2008 Søren Hauberg
## 
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

## -*- texinfo -*-
## @deftypefn {Function File} {@var{E} =} entropyfilt (@var{im})
## @deftypefnx{Function File} {@var{E} =} entropyfilt (@var{im}, @var{domain})
## @deftypefnx{Function File} {@var{E} =} entropyfilt (@var{im}, @var{domain}, @var{padding}, ...)
## Computes the local entropy in a neighbourhood around each pixel in an image.
##
## The entropy of the elements of the neighbourhood is computed as
##
## @example
## @var{E} = -sum (@var{P} .* log2 (@var{P})
## @end example
##
## where @var{P} is the distribution of the elements of @var{im}. The distribution
## is approximated using a histogram with @var{nbins} cells. If @var{im} is
## @code{logical} then two cells are used. For other classes 256 cells
## are used.
##
## When the entropy is computed, zero-valued cells of the histogram are ignored.
##
## The neighbourhood is defined by the @var{domain} binary mask. Elements of the
## mask with a non-zero value are considered part of the neighbourhood. By default
## a 9 by 9 matrix containing only non-zero values is used.
##
## At the border of the image, extrapolation is used. By default symmetric
## extrapolation is used, but any method supported by the @code{padarray} function
## can be used. Since extrapolation is used, one can expect a lower entropy near
## the image border.
##
## @seealso{entropy, paddarray, stdfilt}
## @end deftypefn

function retval = entropyfilt (I, domain = true (9), padding = "symmetric", varargin)
  ## Check input
  if (nargin == 0)
    error ("entropyfilt: not enough input arguments");
  endif
  
  if (!ismatrix (I))
    error ("entropyfilt: first input must be a matrix");
  endif
  
  if (!ismatrix (domain))
    error ("entropyfilt: second input argument must be a logical matrix");
  endif
  domain = (domain > 0);
  
  ## Get number of histogram bins
  if (islogical (I))
    nbins = 2;
  else
    nbins = 256;
  endif
  
  ## Convert to 8 or 16 bit integers if needed
  switch (class (I))
    case {"double", "single", "int16", "int32", "int64", "uint16", "uint32", "uint64"}
      min_val = double (min (I (:)));
      max_val = double (max (I (:)));
      if (min_val == max_val)
        retval = zeros (size (I));
        return;
      endif
      I = (double (I) - min_val)./(max_val - min_val);
      I = uint8 (255 * I);
    case {"logical", "int8", "uint8"}
      ## Do nothing
    otherwise
      error ("entropyfilt: cannot handle images of class '%s'", class (I));
  endswitch
  size (I)
  ## Pad image
  pad = floor (size (domain)/2);
  I = padarray (I, pad, padding, varargin {:});
  even = (round (size (domain)/2) == size (domain)/2);
  idx = cell (1, ndims (I));
  for k = 1:ndims (I)
    idx {k} = (even (k)+1):size (I, k);
  endfor
  I = I (idx {:});
  size (I)
  ## Perform filtering
  retval = __spatial_filtering__ (I, domain, "entropy", I, nbins);

endfunction