% Finds the shape and phase for rate of activity of neighbouring Ganglion cells for 'run' number of trials

function Gaussians(run)

% Set initial variables

[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2+1:0.5:0;
postime = 1:0.5:time/2;
rtime = horzcat(-time/2+0.5,negtime,0.5,postime);
%negtime = -time/2:0.5:0;
%postime = 1:0.5:time/2;
%rtime = horzcat(negtime,postime);

rtime = rtime./1000;
negtime_small = -time/2+1:1:0;
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

freq = zeros(1,run);
FWHM = zeros(1,run);
imaxindex = zeros(1,run);

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
 
end %for get data

%---------------------------------------------------------------------------------------Plotting Gaussians------------------------------------------------------------------------------


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

gwave_nDC(m,:) = gwave(m,:) - mean(gwave(m,:));
gwave_hat(m,:) = fft(gwave_nDC(m,:));

% Find the initial values for the sine fit

[imax,imaxindex(m)] = max(abs(gwave_hat(m,1:length(gwave_hat(1,:)/2))));

freq(m) = (imaxindex(m)-1)/time;

if freq(m) > 120
   freq(m) = freq(m)/2;
elseif  freq(m) < 40
   freq(m) = freq(m)*2;
end %if

B = 2*pi*freq(m);
phase = angle(gwave_hat(m,imaxindex(m)));


% if phase < 0
%    phase = phase + pi;
% elseif phase > 2
%     phase = phase - pi;
% end %if

phaseg_all(m) = phase;

% Return to original data (graph starting at 0) to find amplitude and offset

gwave_half = gwave(m,1:length(N1_A));
amp = (max(gwave_half)-min(gwave_half))/2;
offset = (max(gwave_half)+min(gwave_half))/2;


%Create a plot for each trial with sine fit for half of the points

%theta = [amp,B,phase,offset]; % start with correct values, since we know them
%control = {1000; 1; 1};
%[theta, obj_value, iterations, convergence] = bfgsmin("sinmodelnz", {theta, gwave_half, rtime_half}, control);
%z = theta(1)*cos(theta(2)*rtime_half+theta(3))+theta(4);

z = amp*cos(B*rtime_half+phase) + offset;

plot(rtime_half,z,'r');

totmax = max(N_A(m,:));
%axis([-time/2 time/2 0.1 totmax*1.5]);
ylabel('# of Spikes');
set(gca,'XTick',[]);

title(horzcat('Trial  ',num2str(m)));
box off;


phaseg = num2str(phase);
phase_label = 'phase=';
phase_value = text(0-(time/6),totmax*1.1,strcat(phase_label,phaseg));
set(phase_value,'color',[0 0.6 0],'fontsize',20-1.2*run);

  if m == 1
     legend('Gaussians of data','Cosine fit','Location','NorthEast')
  end %if

end % for run

xlabel('Time (s)')
xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));

hold off

% Plot Fourier transform for all trials w/ Gaussian fit of peak and display values

figure(4);
clf;

fullfourier = sum(abs(gwave_hat),1);

b = 1:length(fullfourier);
b = b./time;

plot(b(1:length(b)/4),fullfourier(1:length(fullfourier)/4));

hold on

[fouriermax,fouriermaxindex] = max(fullfourier);
FWHMfourier = fwhm(fullfourier(fouriermaxindex-4:fouriermaxindex+4));

fsig = FWHMfourier/(2*sqrt(2*log(2)));
fgau = normpdf(b(1:length(b)/4),fouriermaxindex/time,fsig);
fgau = fgau./max(fgau);
fgau = fgau.*(fullfourier(fouriermaxindex));

plot(b(1:length(b)/4),fgau,'r');
xlabel('Frequency (Hz)');
ylabel('Spikes/s');
title('Fourier Transform of Gaussians');
box off;

frequency = num2str((fouriermaxindex-1)/time);
frequency_label = 'Frequency=';
frequency_value = text((fouriermaxindex/time)*2,(4/5)*fouriermax,strcat(frequency_label,frequency));
set(frequency_value,'color',[0 0.6 0],'fontsize',12);

fsig = num2str(fsig);
fsig_label = '\sigma=';
fsig_value = text((fouriermaxindex/time)*2,(3/5)*fouriermax,strcat(fsig_label,fsig));
set(fsig_value,'color',[0 0.6 0],'fontsize',12);

FWHMstr = num2str(FWHMfourier);
FWHM_label = 'FWHM=';
FWHM_value = text((fouriermaxindex/time)*2,(2/5)*fouriermax,strcat(FWHM_label,FWHMstr));
set(FWHM_value,'color',[0 0.6 0],'fontsize',12);

%-------------------------------------------------------Morlets-------------------------------------------------


figure(2);
clf;

for m = 1:run
FWHM(m) = fwhm(abs(gwave_hat(m,imaxindex(m)-5:imaxindex(m)+5)));
end %for run

ind = find(N_A(1,1:length(SumN)/2)); % Get numcols of phase

phase = zeros(run,length(ind));
mwave = zeros(run,length(N1_A));

for m = 1:run

N_A_nDC(m,:) = N_A(m,:) - mean(N_A(m,:));
N_A_hat(m,:) = fft(N_A_nDC(m,:));

% Set bandwith and center frequency parameters

fb = (1/(FWHM(m)*2*pi))^2;
fc = freq(m);

% Set support and grid parameters

lb = rtime(1);
ub = rtime(length(rtime)/2);
n = time*1000;
x = linspace(lb,ub,n);
ind = find(N_A(m,1:length(N_A)/2));
wav = zeros(length(ind),length(N1_A));


for j = 1:length(ind)

displ = (time/2-0.001*ind(j));
wav(j,:) = cmorwavf(lb,ub,n,fb,fc,displ);
mwave(m,:) = mwave(m,:) + wav(j,:);

phase(m,j) = angle(wav(j,length(rtime)/2));

end %for j wav 

mwave(m,:) = mwave(m,:)./max(mwave(m,:));

 % If you want to plot all CMWs as one line

%if m == 3
%figure(7);
%clf;
%plot(rtime_small,mwave(3,:),'r',rtime,gwave(3,:),'b');

end %if
%plot(rtime,wav(1,:),'r',rtime,gau(1,:),'b')
%hold on
%end %for d
%figure(7);
%clf;

x = -pi:pi/10:pi;

xx = -pi+pi/20:pi/10:pi;

phase_dist = histc(phase(m,:),x); % Histogram of phases
phase_dist = (phase_dist./sum(phase_dist))*(10/pi); % Normalize phase_dist
phase_dist = phase_dist(1:length(phase_dist)-1);


if run < 5
r = ceil(run/5);
else r = floor(run/5);
end %if

subplot(r,run/r,m);


h = scatter(xx,phase_dist,7,'b','*');

% Find minimized Von Mises Distribution for phase_dist

theta = [0,1];
control = {1000; 0; 1; 1; 0; 1e-10; 9e-5};
[theta, obj_value, iterations, convergence] = bfgsmin("mises", {theta, phase_dist', xx'}, control);

convergence

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

if k < 0
mu = mu - pi;
end %if

[maxmises,maxmisesindex] = max(mises_result);

phaseofmax(m) = xx(maxmisesindex);
 
plot(xx,mises_result,'r');
box off;

hold on

phasem_all(m) = mu;

phasem = num2str(mu);
phase_label = 'phase=';
phase_value = text((-2/3)*pi,max(mises_result)/2,strcat(phase_label,phasem));
set(phase_value,'color',[0 0.6 0],'fontsize',20-1.2*run);
 if m == 1
    leg = legend('Center of Hist. Bins','Von Mises distribution','Location','north');
    set(leg,'fontsize',20-1.2*run);
    scatter(pi,1.05*max(phase_dist),20-run,'b','*');
    text(-pi,1.05*max(phase_dist),'Center of Hist. Bins','fontsize',20-1.3*run)
 end %if legend
title(horzcat('Trial  ',num2str(m)));
xlabel('Phase (rad)')

axis([-pi pi 0 max(phase_dist)*1.2])

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

keyboard

%hold off

%max_wav = max(real(wav));
%totmax = max(max_wav);

%axis([-time/2 time/2 -totmax*7 totmax*7]);


%xlabel('Time (s)');
%xneg = -time/2:0.1:0;
%xpos = 0:0.1:time/2;
%set(gca,'XTick',horzcat(xneg,xpos));
















%-----------------------------------------------------Plot Gaussians phase vs. CMW phase << disabled-----------------------------------

% Find average difference between Gaussian phase and CMW phase and plot them

   % Sort CMW phases in ascending order and sort corresponding Gaussians accordingly

asc_phasem_all = zeros(1,run);
corr_phaseg_all = zeros(1,run);

[asc_phasem_all,locs] = sort(phasem_all);

for ix = 1:length(locs)
    corr_phaseg_all(ix) = phaseg_all(locs(ix));
end %for sort g's accordingly

    % Plot them

%figure(3);
%clf;
%shg;
%scatter(asc_phasem_all,corr_phaseg_all,7,'b','*'); <================= disabled
hold on
x = -pi:0.01:pi;
y = x;
%plot(x,y,'r');   <============= disabled
%hold on

%box off;


%legend('Differences','Perfect Correlation','Location','SouthEast');
%scatter(pi+pi/5,-pi-pi/6,20,'b','*');
%text(pi/2,-pi-pi/6,'Differences','fontsize',10)

%axis square;

    % Compute avg. difference

differ_tot = 0;
differ = zeros(1,run);
for m = 1:run

differ(m) = abs(corr_phaseg_all(m) - asc_phasem_all(m));
differ_tot = differ_tot + differ(m);

end % for differ

differ_avg = differ_tot/run;

difference = num2str(differ_avg);
%differ_label = 'avg. diff.=';
%differ_value = text(pi/3,pi/3,strcat(differ_label,difference));
%set(differ_value,'color',[0 0.6 0],'fontsize',20);
%title('Correlation of Gaussian phase and CMW phase');

%xlabel('CMW phase');
%ylabel('Gaussian phase');

%figure(9);
%clf;


%hist(differ);   <================ disabled
%axis([-pi pi 0 4])
%set(gca,'YTick',1:max(hist(differ)));
%xlabel('Difference');
%ylabel('Difference');
%title('Difference For Each Trial');
