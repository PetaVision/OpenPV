clear all;

time_series = rand(1, 64) + cos(16 * 2*pi*([0:63]/64));
plot(time_series);
%Fs = 1000;
%T = 1/Fs;
%L = .2;
%t = [0:T:L];
%
%display(t);
%freq = .;
%Amp = 1;
%wave = Amp * sin(2 * pi * freq * t);
%plot(t(1:100), wave(1:100));
%
ft_series = fft(time_series);
figure;
plot(abs(ft_series));

