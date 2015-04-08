% Given N stacked layers, returns the number of pixels in (a neuron in the Nth layer)'s receptive field IN ONE DIRECTION.
%  -Wesley Chavez  04/08/15
% 
% 
%  Example for 4 layers:
%                                                        nxp: The number of neurons in the above layer      stride: The number of neurons in the above layer
%                                                             that each neuron in this layer can see                divided by the number of neurons in this layer
%
%       1                         14
%     o o o o o o o o o o o o o o o o     Layer 1        N/A                                                N/A
%        \       /       \       /         
%         \     /         \     /
%          \   /           \   /
%           \ /             \ /
%            o   o   o   o   o            Layer 2        6                                                  2
%             \     / \     /
%              \  /     \ /
%            o   o   o   o   o            Layer 3        3                                                  1
%                 \     / 
%                   \ / 
%                    o                    Layer 4        3                                                  N/A
%
%
%  nxp = [6 3 3];
%  stride = [2 1];
%  howManyPixelsAmILookingAt(nxp,stride)
%  ans = 14

function numPixels = howManyPixelsAmILookingAt(nxp,stride)
if (size(nxp,2) == 2)
   numPixels = nxp(1) + stride(1)*(nxp(2)-1);
else
   numPixels = nxp(1) + stride(1)*(howManyPixelsAmILookingAt(nxp(2:end),stride(2:end))-1);
end
end
