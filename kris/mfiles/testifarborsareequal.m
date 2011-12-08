close all
clear all
clc;

w0_a0=readweights('/Users/kpeterson/Documents/workspace/kris/output/testarbors/w1_a0_last.pvp');
w0_a1=readweights('/Users/kpeterson/Documents/workspace/kris/output/testarbors/w1_a1_last.pvp');
w0_a2=readweights('/Users/kpeterson/Documents/workspace/kris/output/testarbors/w1_a2_last.pvp');
w0_a3=readweights('/Users/kpeterson/Documents/workspace/kris/output/testarbors/w1_a3_last.pvp');
w0_a4=readweights('/Users/kpeterson/Documents/workspace/kris/output/testarbors/w1_a4_last.pvp');


[x y f blob] = size(w0_a0);
prob=0;
eq=1;
for (xx=[1:x])
    for yy=[1:y]
        for ff=[1:f]
            for bb=[1:blob]
                if(((w0_a0{1}(xx,yy,ff,bb) ~= (w0_a1{1}(xx,yy,ff,bb)))||...
                    (w0_a1{1}(xx,yy,ff,bb) ~= (w0_a2{1}(xx,yy,ff,bb)))||...
                    (w0_a2{1}(xx,yy,ff,bb) ~= (w0_a3{1}(xx,yy,ff,bb)))||...
                    (w0_a3{1}(xx,yy,ff,bb) ~= (w0_a4{1}(xx,yy,ff,bb))))&&...
                    ((isnan(w0_a0{1}(xx,yy,ff,bb))==0)||(isnan(w0_a1{1}(xx,yy,ff,bb))==0)||...
                    (isnan(w0_a2{1}(xx,yy,ff,bb))==0)||(isnan(w0_a3{1}(xx,yy,ff,bb))==0)||(isnan(w0_a4{1}(xx,yy,ff,bb))==0)))
                    eq=0
                    w0_a0{1}(xx,yy,ff,bb)
                    w0_a1{1}(xx,yy,ff,bb)
                    w0_a2{1}(xx,yy,ff,bb)
                    w0_a3{1}(xx,yy,ff,bb)
                    w0_a4{1}(xx,yy,ff,bb)
                    xx
                    yy
                    ff
                    bb
                    prob=prob+1;
                end
            end
        end
    end
end
prob
eq