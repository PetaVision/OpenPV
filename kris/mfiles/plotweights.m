clear all;
close all;
clc;

%a=readweights('~/Documents/workspace/kris/output/krisplay/w2_last.pvp');
a11=readweights('~/Documents/workspace/kris/output/testinitweights/w0_last.pvp');
b=squeeze(a11{1}(4,4,1,:));

x=256;
y=256;

for (i=[1:x])
    for (n=[1:y])
        t1=(i-1)*y
        t2=t1+n
        w(n,i)=b(t2);
    end
end

surf(w)
%ylim([0, 110])

