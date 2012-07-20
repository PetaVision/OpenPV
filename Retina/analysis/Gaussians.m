% Finds the shape and phase for rate of activity of neighbouring Ganglion cells for 'run' number of trials

function Gaussians(run)

% Set initial variables

[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2:0.5:0;
postime = 1:0.5:time/2;
rtime = horzcat(negtime,postime);
rtime = rtime./1000;
negtime_small = (-time+1)/2:1:0;
postime_small = 1:1:time/2;

rtime_small = horzcat(negtime_small,postime_small);
rtime_small = rtime_small./1000;

rtime_half =  rtime(1:length(rtime)/2);
gwave_nDC =  zeros (run,length(N1_A)*2);
gwave_hat = zeros (run,length(N1_A)*2);

mwave_nDC =  zeros (run,length(N1_A)*2);
mwave_hat = zeros (run,length(N1_A)*2);
phase = zeros (1,run);
phaseg_all = zeros(1,run);
phasem_all = zeros(1,run);

N_Abig = zeros (run,9,length(N1_A));

sig = 0.001;
LineLength = 0.1;

N_A = zeros (run,length(N1_A));
N_A_nDC =  zeros (run,length(N1_A));
N_A_hat = zeros (run,length(N1_A));




figure(1);
clf;

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
time = time/1000;

N_Abig(m,1,:) = N1_A;
N_Abig(m,2,:) = N2_A;
N_Abig(m,3,:) = N3_A;
N_Abig(m,4,:) = N4_A;
N_Abig(m,5,:) = N5_A;
N_Abig(m,6,:) = N6_A;
N_Abig(m,7,:) = N7_A;
N_Abig(m,8,:) = N8_A;
N_Abig(m,9,:) = Gin_A;
 
%---------------------------------------------------------------------------------------Plotting Gaussians------------------------------------------------------------------------------




end %for get data

figure(1);
clf;

for m = 1:run

ind = find(N_A(m,:));
gau = zeros(length(ind),time*2000);  
lam = zeros(1,length(ind));
Max = zeros(1,length(ind));
gwave = zeros(run,2*length(N1_A));

subplot(run,1,m);

for j = 1:length(ind)

lam(j) = -time/2+0.001*ind(j);
gau(j,:) = normpdf(rtime,lam(j),sig);
Max(j) = max(gau(j,:));
gau(j,:) = (gau(j,:)./Max(j))*N_A(m,ind(j));
gwave(m,:) = gwave(m,:) + gau(j,:);

end %for j

plot(rtime,gwave(m,:),'b');

hold on

%-------------------------------------------------------------------------Fit sine to and find phase of Gaussians---------------------------------------------------------------------- 

gwave_nDC(m,:) = gwave(m,:) - mean(gwave(m,1:length(N1_A)));
gwave_hat(m,:) = fft(gwave_nDC(m,:));

% Find the initial values for the sine fit

[imax,imaxindex] = max(abs(gwave_hat(m,:)));

freq = (imaxindex-1)/time

if freq > 120
   freq = freq/2;
elseif  freq < 40
   freq = freq*2;
end %if

B = 2*pi*freq;
phase = angle(gwave_hat(m,imaxindex));
  if phase < 0
     phase = phase + pi/2;
  end %if
phaseg_all(m) = phase;

% Return to original data (graph starting at 0) to find amplitude and offset

gwave_half = gwave(m,1:length(N1_A));
amp = (max(gwave_half)-min(gwave_half))/2;
offset = (max(gwave_half)+min(gwave_half))/2;


%Create a plot for each trial with sine fit for half of the points

theta = [amp,B,phase,offset]; % start with correct values, since we know them
control = {1000; 1; 1};
[theta, obj_value, iterations, convergence] = bfgsmin("sinmodelnz", {theta, gwave_half, rtime_half}, control);
%z = theta(1)*cos(theta(2)*rtime_half+theta(3))+theta(4);

z = amp*cos(B*rtime_half+phase) + offset;


plot(rtime_half,z,'r');

totmax = max(N_A(m,:));
axis([-time/2 time/2 0.1 totmax*1.5]);
ylabel('Rate (1/s)');
set(gca,'XTick',[]);

title(horzcat('Trial  ',num2str(m)));

phaseg = num2str(phase);
phase_label = 'phase=';
phase_value = text(0-(time/6),totmax*1.3,strcat(phase_label,phaseg));
set(phase_value,'color',[0 0.6 0],'fontsize',20-1.2*run);

  if m == 1
     legend('Gaussians of data','Sine fit','Location','NorthEast')
  end %if

end % for run

xlabel('Time (ms)')
xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));


hold off

%-------------------------------------------------------Morlets w/out varied amps w/ clumps--------------------------------------------------


figure(2);
clf;

for m = 1:run

N_A_nDC(m,:) = N_A(m,:) - mean(N_A(m,:));
N_A_hat(m,:) = fft(N_A_nDC(m,:));

[imax,imaxindex] = max(abs(N_A_hat(m,:)));

freq = (imaxindex-1)/time

if freq > 120
   freq = freq/2;
elseif  freq < 40
   freq = freq*2;
end %if


% Set bandwith and center frequency parameters

fb = 0.002;
fc = freq;

% Set support and grid parameters

lb = rtime(1)-50/1000;
ub = rtime(length(rtime))+50/1000;
n = 800;


ind = find(N_A(m,:));
phase = zeros(run,length(ind));

subplot(run,1,m);

wav = zeros(length(ind),time*2000);
mwave = zeros(run,2*length(N1_A));

for j = 1:length(ind)

displ = (time/2-0.001*ind(j));
wav(j,:) = cmorwavf(lb,ub,n,fb,fc,displ);
phase(m,j) = angle(wav(j,length(rtime)/2));

end %for j wav 

x = -pi:pi/10:pi;

xx = -pi+pi/20:pi/10:pi;

phase_dist = histc(phase(m,:),x); % Histogram of phases
phase_dist = (phase_dist./sum(phase_dist))*(10/pi); % Normalize phase_dist
phase_dist = phase_dist(1:length(phase_dist)-1);

subplot(run,1,m);


plot(xx,phase_dist,'r');

% Find minimized Von Mises Distribution for phase_dist

theta = [0,1];
control = {100; 1; 1};
[theta, obj_value, iterations, convergence] = bfgsmin("mises", {theta, phase_dist', xx'}, control);

mu = theta(1,:);	
k = theta(2,:);
[zeroth,ierr]=besseli(0,k);
     
     if ierr ~= 0
        ierr_st = num2str(ierr);
        bessel_st = 'bessel call return';
        print(strcat(ierr_st,bessel_st));
     end %if

hold on

mises_result = (exp(k*cos(xx-mu)))/(2*pi*zeroth); 

plot(xx,mises_result,'b');

[max_mises,max_mises_index] = (max(mises_result));

phasem_all(m) = xx(max_mises_index); 

phasem = num2str(xx(max_mises_index));
phase_label = 'phase=';
phase_value = text(0-time,max_mises/2,strcat(phase_label,phasem));
set(phase_value,'color',[0 0.6 0],'fontsize',20-1.2*run);
 if m == 1
    legend('Center of hist. bins','Von Mises distribution','Location','Best');
 end %if legend
title(horzcat('Trial  ',num2str(m)));
xlabel('Phase (rad)')

%mwave(m,:) = mwave(m,:)./max(mwave(m,1:length(rtime)/2));

%for j = 1:9

%    for i = 1:time*1000

%       if N_Abig(m,j,i) ~= 0

%         x = [rtime_small(i)-0.0004+0.00002*j,rtime_small(i)-0.0004+0.00002*j];

 %        y = [m*N_Abig(m,j,i)-LineLength,m*N_Abig(m,j,i)+LineLength];

%         line(x,y,'color','b');

%       end %if

%     end %for %time

% end %for %cells

%hold on

%plot(rtime,mwave(m,:),'r');

%mwave_nDC(m,:) = mwave(m,:) - mean(mwave(m,:));
%mwave_hat(m,:) = fft(mwave_nDC(m,:));
%[imax,imaxindex] = max(abs(mwave_hat(m,:)));
%phase = angle(max(mwave_hat(m,imaxindex)));


%m_max = max(mwave);
%totmax = max(m_max);

%phase = num2str(phase)
%phase_label = 'phase=';
%phase_value = text(0-(time/6),0,strcat(phase_label,phase));
%set(phase_value,'color',[0 0.6 0],'fontsize',20-2*run);

end %for runs

%hold off

%max_wav = max(real(wav));
%totmax = max(max_wav);

%axis([-time/2 time/2 -totmax*7 totmax*7]);


%xlabel('Time (s)');
%xneg = -time/2:0.1:0;
%xpos = 0:0.1:time/2;
%set(gca,'XTick',horzcat(xneg,xpos));
















%-----------------------------------------------------Plot Gaussians phase vs. CMW phase-----------------------------------

figure(3);
clf;

plot(phasem_all,phaseg_all,'b');


differ_tot = 0;

for m = 1:run

differ = abs(phaseg_all(m) - phasem_all(m));
differ_tot = differ_tot + differ;

end % for differ

differ_avg = differ_tot/run;

difference = num2str(differ_avg)
differ_label = 'avg. diff.=';
differ_value = text(pi/2-0.1,pi/2,strcat(differ_label,difference));
set(differ_value,'color',[0 0.6 0],'fontsize',20);
title('Correlation of Gaussian phase and CMW phase');

xlabel('CMW phase');
ylabel('Gaussian phase');