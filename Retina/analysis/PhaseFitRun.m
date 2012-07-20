% Finds the shape and phase for rate of activity of neighbouring Ganglion cells for 'run' number of trials

function PhaseFitRun(run)

% Set initial variables

[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2+1:1:0;
postime = 1:1:time/2;
rtime = horzcat(negtime,postime);
rtime_half = rtime(1:time/2);
rtime = rtime./1000;
rtime_half = rtime_half./1000;


N_A = zeros (run,length(N1_A));
N_A_nDC =  zeros (run,length(N1_A));
N_A_hat = zeros (run,length(N1_A));



% Get data

for m = 1:run

[time, Gin_A] = pvp_readPointLIFprobeRun(m,"GanglionON",{'A'});
[time, N1_A] = pvp_readPointLIFprobeRun(m,"N1GanglionON",{'A'});
[time, N2_A] = pvp_readPointLIFprobeRun(m,"N2GanglionON",{'A'});
[time, N3_A] = pvp_readPointLIFprobeRun(m,"N3GanglionON",{'A'});
[time, N4_A] = pvp_readPointLIFprobeRun(m,"N4GanglionON",{'A'});
[time, N5_A] = pvp_readPointLIFprobeRun(m,"N5GanglionON",{'A'});
[time, N6_A] = pvp_readPointLIFprobeRun(m,"N6GanglionON",{'A'});
[time, N7_A] = pvp_readPointLIFprobeRun(m,"N7GanglionON",{'A'});
[time, N8_A] = pvp_readPointLIFprobeRun(m,"N8GanglionON",{'A'});

SumN = N1_A + N2_A + N3_A + N4_A + N5_A + N6_A + N7_A + N8_A+Gin_A;
SumN = SumN';
N_A(m,:) = SumN;

%for i = 1:length(SumN)
%    if N_A(m,i) > 1
%       N_A(m,i) = 1;
%    end %if 
%end %for

N_A_nDC(m,:) = N_A(m,:) - mean(N_A(m,:));
N_A_hat(m,:) = fft(N_A_nDC(m,:));

end %for %get data

time = time/1000;



clf;
figure(1);

for m = 1:run

% Find the initial values for the sine fit

[imax,imaxindex] = max(abs(N_A_hat(m,:)));

freq = (imaxindex-1)/time

if freq > 120
   freq = freq/2;
elseif  freq < 40
   freq = freq*2;
end %if

B = 2*pi*freq
phase = angle(max(N_A_hat(m,imaxindex)))

% Return to original data (graph starting at 0) to find amplitude and offset

N_A_half = N_A(m,1:length(SumN)/2);

amp = (max(N_A_half)-min(N_A_half))/2

offset = (max(N_A_half)+min(N_A_half))/2


%Create a plot for each trial with sine fit for half of the points

subplot(run,1,m);

%theta = rand(4,1)

theta = [amp,B,phase,offset]; % start with correct values, since we know them
control = {1000; 1; 1};
[theta, obj_value, iterations, convergence] = bfgsmin("sinmodelnz", {theta, N_A_half, rtime_half}, control);
z = theta(1)*cos(theta(2)*rtime_half+theta(3))+theta(4);


scatter(rtime,N_A(m,:),10,'b','*');

hold on

plot(rtime_half,z,'color','r');


%g = amp*sin(2*B*rtime_half+phase)+offset;

%plot(rtime_half,g,'g');

hold off

max_mtx = max(N_A);
totmax = max(max_mtx);

axis([-time/2 time/2 0.1 totmax*1.5]);
ylabel('Rate (1/s)');
set(gca,'XTick',[]);

phase = num2str(theta(3));
phase_label = 'phase=';
phase_value = text(0-(time/6),totmax*1.3,strcat(phase_label,phase));
set(phase_value,'color',[0 0.6 0],'fontsize',20-2*run);




title(horzcat('Trial  ',num2str(m)));

end %for

xlabel('Time (s)');

xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));




%[psi,x] = cmorwavf()