% y = matlabstringtochararray(s)
%
% If the input is a character array (i.e. ischar(s) returns true), returns
% a copy of the input
% If the input is a matlab string (i.e. isstring(s) returns true), returns
% the char array version of the string.
% If neither isstring(s) nor ischar(s) is true, returns the empty char array
function y = matlabstringtochararray(s)

if ischar(s)
    y = s;
elseif isstring(s)
    y = s.char();
else
    y = '';
end%if
end%function
