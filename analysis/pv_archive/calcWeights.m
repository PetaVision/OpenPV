function [weights] = calcWeights(pos, which)
% Integration of PetaVision layer weight calc function and
% Octave/MATLAB.
%
% Typical usage:
% For a given post-synaptic neuron position, calculates the weight of
% the connections to each pre-synaptic neuron, effectively giving the
% receptive field. (Summary: Set post and vary pre.) i.e.:
% weights = calcWeights([17,16,0]); mesh(weights(1:32,1:32,5))
%
% Also, can set pre and vary post, especially helpful for showing
% the effect of co-circularity/lateral connections, or projective fields.

% Unfortunately need to keep these in synch with the code for now:
NX=64;
NY=64;

if (which==1)
	NO=1;
else
	NO=8
endif

N=NX*NY;
weights(NY,NX,NO)=1;
index=1;

for y = 1:NY
	for x = 1:NX
		for o = 1:NO
			if (which==1)
				% for gabor: (V1/S1)
				% [a,b,c] = mexWeightHarness(i,post,[64,64,0,8*8,3.0,0.3,0.7,4.0,4.0,0.2,1.0]);
				% [a,b,c] = mexWeightHarness([y-1,x-1,o-1],pos,[64,64,1.0,5*5,2.7,0.38,0.0,1.7,0.0,0.3,1.0]);
				[a,b,c] = mexWeightHarness([y-1,x-1,o-1],pos,[64,64,1.0,5*5,2.0,0.38,0.0,1.6,0.0,0.3,1.0]);
			else
				% for cocir
				% weights = calcWeights([17,16,0]); mesh(weights(1:32,1:32,5))
				[a,b,c] = mexWeightHarness(pos, [y-1,x-1,o-1],[0.0, NX,NY,1.0,16*16,22.5,16.0*16.0,8*22.5*22.5,0.33*(0.5/0.125/(1-0.9)),0.0,1.5, 225000]);
			end
			weights(y,x,o)=a;
		end
	end
end

end

% Other useful snippets:
% for y=16:18; for x=15:17; mexWeightHarness([17,16,0], [y,x,1], [1.0, 64, 64, 1.0, 32*32, 22.5, 16.0*16.0, 8*22.5*22.5,1,0,0,1.5,0.001]); end; end;

