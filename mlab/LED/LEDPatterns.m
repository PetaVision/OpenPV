function [LED_patterns, ...
    LED_NUMBER_ndx, LED_NUMBER_str, num_LED_NUMBERS, ...
    LED_LETTER_ndx, LED_LETTER_str, num_LED_LETTERS, ...
    LED_RANDOM_ndx, LED_RANDOM_str, num_LED_RANDOM, ...
    LED_BLANK_ndx, LED_DASH_ndx, ...
    LED_SEMANTIC_ndx, num_LED_SEMANTIC, LED_NONSEMANTIC_ndx, num_LED_NONSEMANTIC] = ...
    LEDPatterns()

global num_LED_segs

LED_patterns = zeros(2^num_LED_segs, num_LED_segs+1);
for i_pat = 1:2^num_LED_segs
    residual_val = i_pat;
    log2_exp = floor( log2(residual_val) );
    while (residual_val ~= 0 )
        LED_patterns( i_pat, log2_exp + 1 ) = 1;
        residual_val = residual_val - 2^log2_exp;
        log2_exp = floor( log2(residual_val) );
    end
end
% LED_patterns = (LED_patterns);

% get LED_patters corresponding to NUMBER targets
LED_NUMBER_ndx(1) = sum( [0 0 1 0 0 1 0] .* 2.^( ( [0 0 1 0 0 1 0] .* (0:num_LED_segs-1) ) ) ); % 1,l
LED_NUMBER_ndx(2) = sum( [1 0 1 1 1 0 1] .* 2.^( ( [1 0 1 1 1 0 1] .* (0:num_LED_segs-1) ) ) ); % 2,Z
LED_NUMBER_ndx(3) = sum( [1 0 1 1 0 1 1] .* 2.^( ( [1 0 1 1 0 1 1] .* (0:num_LED_segs-1) ) ) ); % 3
LED_NUMBER_ndx(4) = sum( [0 1 1 1 0 1 0] .* 2.^( ( [0 1 1 1 0 1 0] .* (0:num_LED_segs-1) ) ) ); % 4
LED_NUMBER_ndx(5) = sum( [1 1 0 1 0 1 1] .* 2.^( ( [1 1 0 1 0 1 1] .* (0:num_LED_segs-1) ) ) ); % 5,S
LED_NUMBER_ndx(6) = sum( [1 1 0 1 1 1 1] .* 2.^( ( [1 1 0 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % 6,G
LED_NUMBER_ndx(7) = sum( [1 0 1 0 0 1 0] .* 2.^( ( [1 0 1 0 0 1 0] .* (0:num_LED_segs-1) ) ) ); % 7
LED_NUMBER_ndx(8) = sum( [1 1 1 1 1 1 1] .* 2.^( ( [1 1 1 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % 8,B
LED_NUMBER_ndx(9) = sum( [1 1 1 1 0 1 1] .* 2.^( ( [1 1 1 1 0 1 1] .* (0:num_LED_segs-1) ) ) ); % 9
LED_NUMBER_ndx(10) = sum( [1 1 1 0 1 1 1] .* 2.^( ( [1 1 1 0 1 1 1] .* (0:num_LED_segs-1) ) ) ); % 0,O
LED_NUMBER_ndx(11) = sum( [0 1 0 0 1 0 0] .* 2.^( ( [0 1 0 0 1 0 0] .* (0:num_LED_segs-1) ) ) ); % 1,l (shifted left)
LED_NUMBER_ndx(12) = sum( [1 1 1 0 0 1 0] .* 2.^( ( [1 1 1 0 0 1 0] .* (0:num_LED_segs-1) ) ) ); % 7 (w northwest segment)
LED_NUMBER_ndx(13) = sum( [1 1 1 1 0 1 0] .* 2.^( ( [1 1 1 1 0 1 0] .* (0:num_LED_segs-1) ) ) ); % 9, q (no bottom segment)

LED_NUMBER_str{1} = ' 1';
LED_NUMBER_str{2} = ' 2';
LED_NUMBER_str{3} = ' 3';
LED_NUMBER_str{4} = ' 4';
LED_NUMBER_str{5} = ' 5';
LED_NUMBER_str{6} = ' 6';
LED_NUMBER_str{7} = ' 7';
LED_NUMBER_str{8} = ' 8';
LED_NUMBER_str{9} = ' 9';
LED_NUMBER_str{10} = ' 0';
LED_NUMBER_str{11} = '1 ';
LED_NUMBER_str{12} = '7 ';
LED_NUMBER_str{13} = '9 ';

num_LED_NUMBERS = length(LED_NUMBER_ndx);


% get LED_patters corresponding to non-ambiguous LETTER distractors
LED_LETTER_ndx(1) = sum( [1 1 1 1 1 1 0] .* 2.^( ( [1 1 1 1 1 1 0] .* (0:num_LED_segs-1) ) ) ); % A
LED_LETTER_ndx(2) = sum( [1 1 0 0 1 0 1] .* 2.^( ( [1 1 0 0 1 0 1] .* (0:num_LED_segs-1) ) ) ); % C
LED_LETTER_ndx(3) = sum( [1 1 0 1 1 0 1] .* 2.^( ( [1 1 0 1 1 0 1] .* (0:num_LED_segs-1) ) ) ); % E
LED_LETTER_ndx(4) = sum( [1 1 0 1 1 0 0] .* 2.^( ( [1 1 0 1 1 0 0] .* (0:num_LED_segs-1) ) ) ); % F
LED_LETTER_ndx(5) = sum( [0 1 1 1 1 1 0] .* 2.^( ( [0 1 1 1 1 1 0] .* (0:num_LED_segs-1) ) ) ); % H
LED_LETTER_ndx(6) = sum( [0 0 1 0 1 1 1] .* 2.^( ( [0 0 1 0 1 1 1] .* (0:num_LED_segs-1) ) ) ); % J
LED_LETTER_ndx(7) = sum( [0 1 0 0 1 0 1] .* 2.^( ( [0 1 0 0 1 0 1] .* (0:num_LED_segs-1) ) ) ); % L
LED_LETTER_ndx(8) = sum( [1 1 1 1 1 0 0] .* 2.^( ( [1 1 1 1 1 0 0] .* (0:num_LED_segs-1) ) ) ); % P
LED_LETTER_ndx(9) = sum( [0 1 1 0 1 1 1] .* 2.^( ( [0 1 1 0 1 1 1] .* (0:num_LED_segs-1) ) ) ); % U
LED_LETTER_ndx(10) = sum( [1 0 1 1 1 1 1] .* 2.^( ( [1 0 1 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % a
LED_LETTER_ndx(11) = sum( [0 1 0 1 1 1 1] .* 2.^( ( [0 1 0 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % b
LED_LETTER_ndx(12) = sum( [1 1 0 1 0 0 0] .* 2.^( ( [1 1 0 1 0 0 0] .* (0:num_LED_segs-1) ) ) ); % c, top postion
LED_LETTER_ndx(13) = sum( [0 0 0 1 1 0 1] .* 2.^( ( [0 0 0 1 1 0 1] .* (0:num_LED_segs-1) ) ) ); % c, bottom postion
LED_LETTER_ndx(14) = sum( [0 0 1 1 1 1 1] .* 2.^( ( [0 0 1 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % d
LED_LETTER_ndx(15) = sum( [1 1 1 1 1 0 1] .* 2.^( ( [1 1 1 1 1 0 1] .* (0:num_LED_segs-1) ) ) ); % e
LED_LETTER_ndx(16) = sum( [0 1 0 1 1 1 0] .* 2.^( ( [0 1 0 1 1 1 0] .* (0:num_LED_segs-1) ) ) ); % h
LED_LETTER_ndx(17) = sum( [1 1 1 1 0 0 0] .* 2.^( ( [1 1 1 1 0 0 0] .* (0:num_LED_segs-1) ) ) ); % o, top position
LED_LETTER_ndx(18) = sum( [0 0 0 1 1 1 1] .* 2.^( ( [0 0 0 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % o, bottom position
LED_LETTER_ndx(19) = sum( [1 1 1 0 1 0 0] .* 2.^( ( [1 1 1 0 1 0 0] .* (0:num_LED_segs-1) ) ) ); % r
LED_LETTER_ndx(20) = sum( [0 1 1 1 0 0 0] .* 2.^( ( [0 1 1 1 0 0 0] .* (0:num_LED_segs-1) ) ) ); % u, top position
LED_LETTER_ndx(21) = sum( [0 0 0 0 1 1 1] .* 2.^( ( [0 0 0 0 1 1 1] .* (0:num_LED_segs-1) ) ) ); % u, bottom position
LED_LETTER_ndx(22) = sum( [0 1 1 1 0 1 1] .* 2.^( ( [0 1 1 1 0 1 1] .* (0:num_LED_segs-1) ) ) ); % y

LED_LETTER_str{1} = ' A ';
LED_LETTER_str{2} = ' C ';
LED_LETTER_str{3} = ' E ';
LED_LETTER_str{4} = ' H ';
LED_LETTER_str{5} = ' J ';
LED_LETTER_str{6} = ' L ';
LED_LETTER_str{7} = ' P ';
LED_LETTER_str{8} = ' U ';
LED_LETTER_str{9} = ' F ';
LED_LETTER_str{10} = ' a ';
LED_LETTER_str{11} = ' b ';
LED_LETTER_str{12} = ' c ';
LED_LETTER_str{13} = 'c  ';
LED_LETTER_str{14} = ' d ';
LED_LETTER_str{15} = ' 3 ';
LED_LETTER_str{16} = ' h ';
LED_LETTER_str{17} = 'o  ';
LED_LETTER_str{18} = ' o ';
LED_LETTER_str{19} = ' r ';
LED_LETTER_str{20} = ' u ';
LED_LETTER_str{21} = 'u  ';
LED_LETTER_str{22} = ' y ';

num_LED_LETTERS = length(LED_LETTER_ndx);


% get LED_patters corresponding to other semantic objects
LED_DASH_ndx = sum( [0 0 0 1 0 0 0] .* 2.^( ( [0 0 0 1 0 0 0] .* (0:num_LED_segs-1) ) ) ); % -
LED_BLANK_ndx = 128; % sum( [0 0 0 0 0 0 0 1] .* 2.^( ( [0 0 0 0 0 0 0 1] .* (0:num_LED_segs) ) ) ); % blank


% get LED_patters corresponding to selected non-semantic distractors
LED_RANDOM_ndx(1) = sum( [0 1 1 1 1 1 1] .* 2.^( ( [0 1 1 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % r1, upside down A
LED_RANDOM_ndx(2) = sum( [1 0 1 0 1 0 1] .* 2.^( ( [1 0 1 0 1 0 1] .* (0:num_LED_segs-1) ) ) ); % r2, 2 with missing center
LED_RANDOM_ndx(3) = sum( [1 1 1 0 1 0 1] .* 2.^( ( [1 1 1 0 1 0 1] .* (0:num_LED_segs-1) ) ) ); % r3, 0 with missing southeast corner
LED_RANDOM_ndx(4) = sum( [1 1 1 1 0 0 1] .* 2.^( ( [1 1 1 1 0 0 1] .* (0:num_LED_segs-1) ) ) ); % r4, craig bad hair day
LED_RANDOM_ndx(5) = sum( [0 0 1 1 0 1 1] .* 2.^( ( [0 0 1 1 0 1 1] .* (0:num_LED_segs-1) ) ) ); % r5, upside down F
LED_RANDOM_ndx(6) = sum( [1 1 1 0 0 1 1] .* 2.^( ( [1 1 1 0 0 1 1] .* (0:num_LED_segs-1) ) ) ); % r6, 0 with missing southwest corner
LED_RANDOM_ndx(7) = sum( [1 0 0 1 1 1 1] .* 2.^( ( [1 0 0 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % r7, 0 with missing northeast corner
LED_RANDOM_ndx(8) = sum( [1 1 0 0 1 1 1] .* 2.^( ( [1 1 0 0 1 1 1] .* (0:num_LED_segs-1) ) ) ); % r8, 0 with missing northwest corner
LED_RANDOM_ndx(9) = sum( [1 0 0 1 1 1 1] .* 2.^( ( [1 0 0 1 1 1 1] .* (0:num_LED_segs-1) ) ) ); % r9, upside down craig bad hair day
LED_RANDOM_ndx(10) = sum( [1 1 0 0 0 1 1] .* 2.^( ( [1 1 0 0 0 1 1] .* (0:num_LED_segs-1) ) ) ); % r10, 5 with missing center

LED_RANDOM_str{1} = 'r1 ';
LED_RANDOM_str{2} = 'r2 ';
LED_RANDOM_str{3} = 'r3 ';
LED_RANDOM_str{4} = 'r4 ';
LED_RANDOM_str{5} = 'r5 ';
LED_RANDOM_str{6} = 'r6 ';
LED_RANDOM_str{7} = 'r7 ';
LED_RANDOM_str{8} = 'r8 ';
LED_RANDOM_str{9} = 'r9 ';
LED_RANDOM_str{10} = 'r10';

num_LED_RANDOM = length(LED_RANDOM_ndx);


LED_SEMANTIC_ndx = [ LED_NUMBER_ndx, LED_LETTER_ndx];
num_LED_SEMANTIC = length(LED_SEMANTIC_ndx);

LED_ndx = 1:((2^num_LED_segs)+1);
LED_NONSEMANTIC_ndx = LED_ndx;
LED_NONSEMANTIC_ndx(LED_SEMANTIC_ndx) = 0;
LED_NONSEMANTIC_ndx = LED_NONSEMANTIC_ndx(LED_NONSEMANTIC_ndx ~= 0);
num_LED_NONSEMANTIC = length(LED_NONSEMANTIC_ndx);


