function Samples = getSamples(max, percentage)
Samples = find(rand(1, max) < percentage);
endfunction
