close all;
clear all;
clc;


x=[1:4];
y=[1:256];
aspect=1/64;

for(i=[1:4])
    for(n=[1:256])
        z(i,n)=exp(-((x(i)-2)*(x(i)-2)+(aspect^2)*(y(n)-128)*(y(n)-128))/(4^2));
    end
end

surf(z)