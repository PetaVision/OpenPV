function fh = plotLEDPattern( LED_pattern)

global num_LED_segs

% define individual LED segments
gap = 0.05; 
min_x = 0;
max_x = 1;
min_y = 0;
max_y = 2;
mid_y = 1;

x = 1;
y = 2;
start = 1;
finish = 2;
north = 1;
northwest = 2;
northeast = 3;
center = 4;
southwest = 5;
southeast = 6;
south = 7;

LED_segment(north, start, x) = min_x + gap;
LED_segment(north, start, y) = max_y;
LED_segment(north, finish, x) = max_x - gap;
LED_segment(north, finish, y) = max_y;

LED_segment(northwest, start, x) = min_x;
LED_segment(northwest, start, y) = mid_y + gap;
LED_segment(northwest, finish, x) = min_x;
LED_segment(northwest, finish, y) = max_y - gap;

LED_segment(northeast, start, x) = max_x;
LED_segment(northeast, start, y) = mid_y + gap;
LED_segment(northeast, finish, x) = max_x;
LED_segment(northeast, finish, y) = max_y - gap;

LED_segment(center, start, x) = min_x + gap;
LED_segment(center, start, y) = mid_y;
LED_segment(center, finish, x) = max_x - gap;
LED_segment(center, finish, y) = mid_y;

LED_segment(southwest, start, x) = min_x;
LED_segment(southwest, start, y) = min_y + gap;
LED_segment(southwest, finish, x) = min_x;
LED_segment(southwest, finish, y) = mid_y - gap;

LED_segment(southeast, start, x) = max_x;
LED_segment(southeast, start, y) = min_y + gap;
LED_segment(southeast, finish, x) = max_x;
LED_segment(southeast, finish, y) = mid_y - gap;

LED_segment(south, start, x) = min_x + gap;
LED_segment(south, start, y) = min_y;
LED_segment(south, finish, x) = max_x - gap;
LED_segment(south, finish, y) = min_y;

fh = figure;
axis off
box off
axis( [ (min_x - gap) (max_x + gap) (min_y - gap) (max_y + gap) ] );
for i_seg = 1 : num_LED_segs
  if LED_pattern(i_seg)
    lh = line( [ LED_segment(i_seg, start, x) LED_segment(i_seg, finish, x) ] , ...
	      [ LED_segment(i_seg, start, y) LED_segment(i_seg, start, y) ] );
  end
end
