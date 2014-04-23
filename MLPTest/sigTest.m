clear all
close all
alpha = .1 %Adjusts the shape of the sigmoid. 1=piecewise.
vthrest = .2 %Shifts x axis over 
vrest = -.2 

vals = -1:.05:1;

vth = (vthrest+vrest)/2;
sig_scale = -log(1./alpha-1)./(vth - vrest);
a = 1./(1 + exp(2 .* (vals - vth) .* sig_scale));

%Derivative
der = -.5 .* sig_scale .* (1./((cosh(sig_scale .* (vth - vals)) .^ 2)));
%num = 2*sig_scale.*exp(2.*sig_scale.*(vals - vth));
%dem = (1+exp(2.*sig_scale.*(vals-vth))).^2;
%der = -num./dem;
%
figure
hold on
plot(vals, a);
plot(vals, der);
hold off
axis([-1 1 -1 1]);
