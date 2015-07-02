# Either converts vector to matrix, row first:
#
#                        a b c
# a b c d e f g h i -->  d e f
#                        g h i
#
#
# Or returns dimension of 'squarest' factors
#
# 15 -> (5,3)


function out = squarify(in)
  if size(in)(1) && size(in)(2) == 1
    n = in;
  else
    n = max(size(in));
  endif
  mark = floor(sqrt(double(n)));
  nearest_square = double(n);
  while (floor(n/mark) != n/mark)
    mark = mark-1;
  endwhile
  fact1 = n/mark;
  fact2 = mark;

  if size(in)(1) && size(in)(2) == 1
    out = [fact1, fact2];
  else
    out = reshape(in, [fact1, fact2])';
  endif
endfunction
