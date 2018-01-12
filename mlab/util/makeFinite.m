% Use this function with modifyEach. Replaces
% any NaNs or Infs in a matrix with zeros.

function result = makeFinite(data)

   data(isnan(data)) = 0;
   data(~isfinite(data)) = 0;
   result = data;

end
