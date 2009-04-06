function [A B phi r] = setSegment(S, nrows, i)
    
    A = repmat(S.A,[nrows 1]);
    B = repmat(S.B,[nrows 1]);

    if nargin == 3
        phi = repmat((S.length*(i):S.length*(i)+S.delta_length(i)-1)' * ...
            S.delta_phi, [1 S.N]) .* repmat(1:S.N,[S.delta_length(i) 1]);
    
    else
        phi = repmat((1:nrows)' * S.delta_phi, [1 S.N]) .* repmat((1:S.N),[nrows 1]);
    
    end

    r = S.A_0 + sum((A .* cos(phi)) + (B .* sin(phi)),2);

end


            