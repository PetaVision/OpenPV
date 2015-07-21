function isoct = isOctave()

	if exist('OCTAVE_VERSION')
		isoct = 1;
	else
		isoct = 0;
	end
end
