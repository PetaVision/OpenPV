function commands

  global w0

  addpath('/Applications/Psychtoolbox/');

% number of targets/fourier component
  numT = 1000;
  screen_color = [];
				%screen_rect = [0 0 256 256];
  screen_rect = [0 0 128 128];
[w0, window_rect]  = Screen('OpenWindow', 0, screen_color, screen_rect);
Screen('FillRect', w0, GrayIndex(w0));

				%fourC = [2 4 6 8];
fourC = [4];

for i = 1:length(fourC)
    nfour = fourC(i);
    for j = 1:numT
	getAmoebaStats2(j, nfour)
	disp(num2str(j));
    end
end

Screen('CloseAll');