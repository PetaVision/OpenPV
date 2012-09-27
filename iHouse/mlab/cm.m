function [cm] = cm()
   cm = ones(100, 3);
   vec = (0:1/49:1)';
   udvec = flipud(vec);

   cm(1:50, 2) = vec;
   cm(1:50, 3) = vec;
   cm(51:100, 1) = udvec;
   cm(51:100, 2) = udvec;
end
