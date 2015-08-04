% This is a script file called from "icaplot.m"
1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotcompare(s1, n1, s2, n2, range, xrange);
  style=getStyles;
  K = regress(s1(n1,:)',s2');
  plot(xrange, s1(n1,range), char(style(1)));
  hold on
  for i=1:size(n2,2)
    plotstyle=char(style(i+1));
    plot(xrange, K(n2(i))*s2(n2(i),range), plotstyle);
  end
  hold off
endfunction

function [legendText, legendStyle]=legendcompare(n1, n2, s1l, s2l, externalLegend);
  style=getStyles;
  if (externalLegend)
    legendText(1)=[s1l ' (see the titles)'];
  else
    legendText(1)=[s1l ' ', int2str(n1)];
  end
  legendStyle(1)=style(1);
  for i=1:size(n2, 2)
    legendText(i+1) = [s2l ' ' int2str(n2(i))];
    legendStyle(i+1) = style(i+1);
  end
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotsum(s1, n1, s2, n2, range, xrange);
  K = diag(regress(s1(n1,:)',s2'));
  sigsum = sum(K(:,n2)*s2(n2,:));
  plot(xrange, s1(n1, range),'k-', ...
       xrange, sigsum(range), 'b-');
endfunction

function [legendText, legendStyle]=legendsum(n1, n2, s1l, s2l, externalLegend);
  if (externalLegend)
    legendText(1)=[s1l ' (see the titles)'];
  else
    legendText(1)=[s1l ' ', int2str(n1)];
  end
  legendText(2)=['Sum of ' s2l ': ', int2str(n2)];
  legendStyle=['k-';'b-'];
endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotsumerror(s1, n1, s2, n2, range, xrange);
  K = diag(regress(s1(n1,:)',s2'));
  sigsum = sum(K(:,n2)*s2(n2,:));
  plot(xrange, s1(n1, range),'k-', ...
       xrange, sigsum(range), 'b-', ...
       xrange, s1(n1, range)-sigsum(range), 'r-');
endfunction

function [legendText, legendStyle]=legendsumerror(n1, n2, s1l, s2l, externalLegend);
  if (externalLegend)
    legendText(1)=[s1l ' (see the titles)'];
  else
    legendText(1)=[s1l ' ', int2str(n1)];
  end
  legendText(2)=['Sum of ' s2l ': ', int2str(n2)];
  legendText(3)='"Error"';
  legendStyle=['k-';'b-';'r-'];
endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function style=getStyles;
  color = ['k','r','g','b','m','c','y'];
  line = ['-',':','-.','--'];
  for i = 0:size(line,2)-1
    for j = 1:size(color, 2)
      style(j + i*size(color, 2)) = strcat(color(j), line(i+1));
    end
  end
endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function range=chkrange(r, s)
  if r == 0
    range = 1:size(s, 2);
  else
    range = r;
  end
endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xrange=chkxrange(xr,r);
  if xr == 0
    xrange = r;
  elseif size(xr, 2) == 2
    xrange = xr(1):(xr(2)-xr(1))/(size(r,2)-1):xr(2);
  elseif size(xr, 2)~=size(r, 2)
    error('Xrange and range have different sizes.');
  else
    xrange = xr;
  end
endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n=chkn(n,s)
  if n == 0
    n = 1:size(s, 1);
  end
endfunction
