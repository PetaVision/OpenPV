function icaplot(mode, varargin);

%ICAPLOT - plot signals in various ways
%
% ICAPLOT is mainly for plottinf and comparing the mixed signals and
% separated ica-signals.
%
% ICAPLOT has many different modes. The first parameter of the function
% defines the mode. Other parameters and their order depends on the
% mode. The explanation for the more common parameters is in the end.
%
% Classic
%     icaplot('classic', s1, n1, range, xrange, titlestr)
%
%     Plots the signals in the same manner as the FASTICA and FASTICAG
%     programs do. All the signals are plotted in their own axis.
%
% Complot
%     icaplot('complot', s1, n1, range, xrange, titlestr)
%
%     The signals are plotted on the same axis. This is good for
%     visualization of the shape of the signals. The scale of the signals 
%     has been altered so that they all fit nicely.
%
% Histogram
%     icaplot('histogram', s1, n1, range, bins, style)
%     
%     The histogram of the signals is plotted. The number of bins can be
%     specified with 'bins'-parameter. The style for the histograms can
%     be either 'bar' (default) of 'line'.
%
% Scatter
%     icaplot('scatter', s1, n1, s2, n2, range, titlestr, s1label,
%     s2label, markerstr)
%
%     A scatterplot is plotted so that the signal 1 is the 'X'-variable
%     and the signal 2 is the 'Y'-variable. The 'markerstr' can be used
%     to specify the maker used in the plot. The format for 'markerstr'
%     is the same as for Matlab's PLOT. 
%
% Compare
%     icaplot('compare', s1, n1, s2, n2, range, xrange, titlestr,
%     s1label, s2label)
%
%     This is for comparing two signals. The main used in this context
%     would probably be to see how well the separated ICA-signals explain 
%     the observed mixed signals. The s2 signals are first scaled with
%     REGRESS function.
%
% Compare - Sum
%     icaplot('sum', s1, n1, s2, n2, range, xrange, titlestr, s1label,
%     s2label)
%
%     The same as Compare, but this time the signals in s2 (specified by
%     n2) are summed together.
%
% Compare - Sumerror
%     icaplot('sumerror', s1, n1, s2, n2, range, xrange, titlestr,
%     s1label, s2label)
%     
%     The same as Compare - Sum, but also the 'error' between the signal
%     1 and the summed IC's is plotted.
%
%
% More common parameters
%     The signals to be plotted are in matrices s1 and s2. The n1 and n2
%     are used to tell the index of the signal or signals to be plotted
%     from s1 or s2. If n1 or n2 has a value of 0, then all the signals
%     from corresponding matrix will be plotted. The values for n1 and n2 
%     can also be vectors (like: [1 3 4]) In some casee if there are more
%     than 1 signal to be plotted from s1 or s2 then the plot will
%     contain as many subplots as are needed. 
%
%     The range of the signals to be plotted can be limited with
%     'range'-parameter. It's value is a vector ( 10000:15000 ). If range 
%     is 0, then the whole range will be plotted.
%
%     The 'xrange' is used to specify only the labels used on the
%     x-axis. The value of 'xrange' is a vector containing the x-values
%     for the plots or [start end] for begin and end of the range
%     ( 10000:15000 or [10 15] ). If xrange is 0, then value of range
%     will be used for x-labels.
%
%     You can give a title for the plot with 'titlestr'. Also the
%     's1label' and 's2label' are used to give more meaningfull label for 
%     the signals.
%
%     Lastly, you can omit some of the arguments from the and. You will
%     have to give values for the signal matrices (s1, s2) and the
%     indexes (n1, n2)

% 7.8.1998

% Added by DR:
% Octave complains if we run icaplot.m as a script file.  So
% call the rest of the functions needed in the following script:

icaplotfunctions;

args = list(all_va_args);

str_param = lower(mode);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'dispsig' is to replace the old DISPSIG
% '' & 'classic' are just another names - '' quite short one :-)

if strcmp(str_param, '') | strcmp(str_param, 'classic') | strcmp(str_param,'dispsig')
  
    % icaplot(mode, s1, n1, range, xrange, titlestr)
    va_start();
    if nargin-1 < 1, error('Not enough arguments.'); end
    s1 = args{1};
    if nargin-1 < 2, n1 = 0; else n1 = args{2}; end
    if nargin-1 < 3, range = 0;else range = args{3}; end
    if nargin-1 < 4, xrange = 0;else xrange = args{4}; end
    if nargin-1 < 5, titlestr = ''; else titlestr = args{5}; end
    range=chkrange(range, s1);
    xrange=chkxrange(xrange, range);
    n1=chkn(n1, s1);
    clg;
    
    numSignals = size(n1, 2);
    for i = 1:numSignals,
	subplot(numSignals, 1, i);
	% Added by DR
	clg;
	% "if" statement added by DR to prevent from trying to plot empty matrices
	if (!all(s1(n1(i),range) == zeros(size(range)))),
	    plot(xrange, s1(n1(i), range)); 
	end
    end
    subplot(numSignals,1, 1);
    if (~isempty(titlestr))
	title(titlestr);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(str_param, 'complot'),
    % icaplot(mode, s1, n1, range, xrange, titlestr)
    va_start();
    if nargin-1 < 1, error('Not enough arguments.'); end
    s1 = remmean(args{1});
    if nargin-1 < 2, n1 = 0;else n1 = args{2}; end
    if nargin-1 < 3, range = 0;else range = args{3}; end
    if nargin-1 < 4, xrange = 0;else xrange = args{4}; end
    if nargin-1 < 5, titlestr = '';else titlestr = args{5}; end
    
    range=chkrange(range, s1);
    xrange=chkxrange(xrange, range);
    n1=chkn(n1, s1);
    
    for i = 1:size(n1, 2)
	S1(i, :) = s1(n1(i), range);
    end
    
    alpha = mean(max(S1')-min(S1'));
    for i = 1:size(n1,2)
	S2(i,:) = S1(i,:) - alpha*(i-1)*ones(size(S1(1,:)));
    end
    
    plot(xrange, S2');
    axis([min(xrange) max(xrange) min(min(S2)) max(max(S2)) ]);
    
    set(gca,'YTick',(-size(S1,1)+1)*alpha:alpha:0);
    set(gca,'YTicklabel',fliplr(n1));
    
    if (~isempty(titlestr))
	title(titlestr);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(str_param, 'histogram'),
    % icaplot(mode, s1, n1, range, bins, style)
    if nargin-1 < 1, error('Not enough arguments.'); end
    s1 = args{1};
    if nargin-1 < 2, n1 = 0;else n1 = args{2}; end
    if nargin-1 < 3, range = 0;else range = args{3}; end
    if nargin-1 < 4, bins = 100;else bins = args{4}; end
    if nargin-1 < 5, style = 'bar';else style = args{5}; end
    
    range = chkrange(range, s1);
    n1 = chkn(n1, s1);
    % Added by DR
    % Make the number of bins a factor of ten different from 
    % the number of data points:
    
    numSignals = size(n1, 2);
    rows = floor(sqrt(numSignals));
    columns = ceil(sqrt(numSignals));
    while (rows * columns < numSignals)
	columns = columns + 1;
    end
    
    str_param = lower(style);
    if strcmp(str_param, 'bar'),
	for i = 1:numSignals,
	    subplot(rows, columns, i);
	    % Added by DR:
	    clg;
	    % "if" statement added by DR to prevent from trying to plot empty matrices
	    if (!all(s1(n1(i),range) == zeros(size(range)))),
		hist(s1(n1(i), range), bins);
	    end  
	    title(int2str(n1(i)));
	    %drawnow;
	    endfor
	    
	elseif strcmp(str_param,''),
	    for i = 1:numSignals,
		subplot(rows, columns, i);
		% Added by DR:
		clg;
		% "if" statement added by DR to prevent from trying to plot empty matrices
		if (!all(s1(n1(i),range) == zeros(size(range)))),
		    hist(s1(n1(i), range), bins);
		end
		title(int2str(n1(i)));
		%drawnow;
	    end
	    
	elseif strcmp(str_param, 'line'),
	    for i = 1:numSignals,
		subplot(rows, columns, i);
		if (!all(s1(n1(i),range) == zeros(size(range)))),
		    [Y, X]=hist(s1(n1(i), range), bins);
		end
		% Added by DR:
		clg;
		%plot(X, Y);
		title(int2str(n1(i)));
		%drawnow;
	    end
	else
	    fprintf('Unknown style.\n')
	end
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif strcmp(str_param, 'scatter'),
	% icaplot(mode, s1, n1, s2, n2, range, titlestr, xlabelstr, ylabelstr, markerstr)
	if nargin-1 < 4, error('Not enough arguments.'); end
	s1 = args{1};
	n1 = args{2};
	s2 = args{3};
	n2 = args{4};
	if nargin-1 < 5, range = 0;else range = args{5}; end
	if nargin-1 < 6, titlestr = '';else titlestr = args{6}; end
	if nargin-1 < 7, xlabelstr = 'Signal 1';else xlabelstr = args{7}; end
	if nargin-1 < 8, ylabelstr = 'Signal 2';else ylabelstr = args{8}; end
	if nargin-1 < 9, markerstr = '.';else markerstr = args{9}; end
	
	range = chkrange(range, s1);
	n1 = chkn(n1, s1);
	n2 = chkn(n2, s2);
	
	rows = size(n1, 2);
	columns = size(n2, 2);
	for r = 1:rows
	    for c = 1:columns
		subplot(rows, columns, (r-1)*columns + c);
		% Added by DR:
		clg;
		plot(s1(n1(r), range),s2(n2(c), range),markerstr);
		if (~isempty(titlestr))
		    title(titlestr);
		end
		if (rows*columns == 1)
		    xlabel(xlabelstr);
		    ylabel(ylabelstr);
		else 
		    xlabel([xlabelstr ' (' int2str(n1(r)) ')']);
		    ylabel([ylabelstr ' (' int2str(n2(c)) ')']);
		end
		%drawnow;
	    end
	end
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif strcmp(str_param, 'compare') | strcmp(str_param, 'sum') | strcmp(str_param,'sumerror'),
	% icaplot(mode, s1, n1, s2, n2, range, xrange, titlestr, s1label, s2label)
	va_start();
	if nargin-1 < 4, error('Not enough arguments.'); end
	s1 = args{1};
	n1 = args{2};
	s2 = args{3};
	n2 = args{4};
	if nargin-1 < 5, range = 0;else range = args{5}; end
	if nargin-1 < 6, xrange = 0;else xrange = args{6}; end
	if nargin-1 < 7, titlestr = '';else titlestr = args{7}; end
	if nargin-1 < 8, s1label = 'Mix';else s1label = args{8}; end
	if nargin-1 < 9, s2label = 'IC';else s2label = args{9}; end
	
	range = chkrange(range, s1);
	xrange = chkxrange(xrange, range);
	n1 = chkn(n1, s1);
	n2 = chkn(n2, s2);
	
	numSignals = size(n1, 2);
	if (numSignals > 1)
	    externalLegend = 1;
	else
	    externalLegend = 0;
	end
	
	rows = floor(sqrt(numSignals+externalLegend));
	columns = ceil(sqrt(numSignals+externalLegend));
	while (rows * columns < (numSignals+externalLegend))
	    columns = columns + 1;
	end
	
	clf;
	
	for j = 1:numSignals
	    subplot(rows, columns, j);
	    str_param = lower(mode);
	    if strcmp(str_param, 'compare')
		plotcompare(s1, n1(j), s2,n2, range, xrange);
		[legendtext,legendstyle]=legendcompare(n1(j),n2,s1label,s2label,externalLegend);
	    elseif strcmp(str_param, 'sum')
		plotsum(s1, n1(j), s2,n2, range, xrange);
		[legendtext,legendstyle]=legendsum(n1(j),n2,s1label,s2label,externalLegend);
	    elseif strcmp(str_param, 'sumerror')
		plotsumerror(s1, n1(j), s2,n2, range, xrange);
		[legendtext,legendstyle]=legendsumerror(n1(j),n2,s1label,s2label,externalLegend);
	    end
	    
	    if externalLegend
		title([titlestr ' (' s1label  ' ' int2str(n1(j)) ')']);
	    else
		legend(char(legendtext));
		if (~isempty(titlestr))
		    title(titlestr);
		end
	    end
	end
	
	if (externalLegend)
	    subplot(rows, columns, numSignals+1);
	    legendsize = size(legendtext, 2);
	    hold on;
	    for i=1:legendsize
		plot([0 1],[legendsize-i legendsize-i], char(legendstyle(i)));
		text(1.5, legendsize-i, char(legendtext(i)));
	    end
	    hold off;
	    axis([0 6 -1 legendsize]);
	    %axis off;
	end
	
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end
    

