clear all
close all
alpha = .1%Adjusts the shape of the sigmoid. 1=piecewise.

vals = -3:.05:3;

%vth = (vthrest+vrest)/2;
%sig_scale = -log(1./alpha-1)./(vth - vrest);
%a = 1./(1 + exp(2 .* (vals - vth) .* sig_scale));
a = 1.7159 .* tanh((2/3).*vals) + alpha .* vals;

%Derivative
%der = -.5 .* sig_scale .* (1./((cosh(sig_scale .* (vth - vals)) .^ 2)));
der = 1.14393 .* (1./((cosh((2/3)*vals)).^2)) + alpha;
%num = 2*sig_scale.*exp(2.*sig_scale.*(vals - vth));
%dem = (1+exp(2.*sig_scale.*(vals-vth))).^2;
%der = -num./dem;
%
figure
hold on
plot(vals, a);
plot(vals, der);
hold off
axis([-3 3 -3 3]);
