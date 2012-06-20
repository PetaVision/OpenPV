close all
clear all
clc

theta = pi/4;
shiftT=2.5;

for(shift=[0:0.25:2])
%shift=1.25;
sigmax=1.25;
%shift=pi/sigmax;
%shift2=5;
sigmat=6;
%shiftT=sigmat/2;
%sigmax=1;
amp=60;
multiplier=0.5;

%for(t=[-20:0.5:20])
for(t=[-9:1:0])
    %for(x=[-20:0.5:20])
    for(x=[-15:1:15])
        tp = t * cos(theta) + x * sin(theta);
        xp = x * cos(theta) - t * sin(theta);
        fx=exp(-(xp^2)/(2*sigmax^2));
        fxp=multiplier*exp(-((xp+shift)^2)/(2*sigmax^2));
        fxm=multiplier*exp(-((xp-shift)^2)/(2*sigmax^2));
        
        %fxp=0; fxm=0;
        %fx=0;
        ft=amp*exp(-((tp+shiftT)^2)/(2*sigmat^2));
        fxt(int32((x+15)*1+1),int32((t+9)*1+1)) = ft * fx - ft * fxp - ft * fxm;
    end
end

t=[-9:1:0];
x=[-15:1:15];


figure; surf(t, x,fxt)
title(['left amp shift ', num2str(shiftT)]);
NFFT=2^nextpow2(length(t));
ftt=linspace(-14.5,14.5,NFFT);
fx=linspace(-5,5,NFFT);
%FFTFXT = fft2(fxt,NFFT, NFFT)/length(t);
FFTFXT = fft2(fxt,NFFT,NFFT)/length(t);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
surf(ftt,fx, abs(fftshift(FFTFXT)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title(['left amp shift ', num2str(shift)]);
view(0,90);
% 
% FFTFXT = fft2(fxt,NFFT,NFFT)/length(t);
% figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
% surf(ftt,fx, angle(fftshift(FFTFXT)))
% %title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
% title('left angle');

end

%for(amp=[32.5:0.1:33])
amp=33;
for(t=[-9:1:0])
    %for(x=[-20:0.5:20])
    for(x=[-15:1:15])
        tp = t * cos(theta) + x * sin(theta);
        xp = x * cos(theta) - t * sin(theta);
        fxpb=exp(-((xp+shift)^2)/(2*sigmax^2));
        fxpl=multiplier*exp(-((xp-2*shift)^2)/(2*sigmax^2));
        fxmb=exp(-((xp-shift)^2)/(2*sigmax^2));
        fxml=multiplier*exp(-((xp+2*shift)^2)/(2*sigmax^2));
        
        %fxp=0; fxm=0;
        %fx=0;
        ft=amp*exp(-((tp+shiftT)^2)/(2*sigmat^2));
        fxtr(int32((x+15)*1+1),int32((t+9)*1+1)) = ft * fxpb - ft * fxmb + ft * fxpl - ft * fxml;
    end
end
t=[-9:1:0];
x=[-15:1:15];
figure; surf(t, x,fxtr)
title(['surf amp = ', num2str(amp)]);
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
FFTFXTr = fft2(fxtr,NFFT,NFFT)/length(t);
surf(ftt,fx, abs(fftshift(FFTFXTr)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title(['right abs amp = ', num2str(amp)]);

%figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
%FFTFXTr = fft2(fxtr,NFFT,NFFT)/length(t);
%surf(ftt,fx, angle(fftshift(FFTFXTr)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
%title(['right angle amp = ', num2str(amp)]);

%end
%figure(1); plot(t, ft);
%figure(2); plot(x, fx);

%fxt = ft' * fx;


figure;
surf(ftt,fx, real(fftshift(FFTFXT)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title('left real');
figure;
surf(ftt,fx, imag(fftshift(FFTFXT)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title('left imag');
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
surf(ftt,fx, real(fftshift(FFTFXTr)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title('right real');
figure; %plot(fx, abs(fftshift(FFTFXT(:,21))));  %
surf(ftt,fx, imag(fftshift(FFTFXTr)))
%title(['shift = ', num2str(shift), ' sigmax = ', num2str(sigmax), ' sigmat = ', num2str(sigmat)]);
title('right imag');
view(45,90);

% i=find(abs(FFTFXT([1:20],20))==max(abs(FFTFXT([1:20],20))))
% if((isempty(i)~=1)&&(length(i)==1))peakloc(shift+1)=i;
% else peakloc(shift+1)=0;
% end
%end