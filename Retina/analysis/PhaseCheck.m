function PhaseCheck(run)

[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

N_A = zeros (run,length(N1_A));
N_A_nDC =  zeros (run,length(N1_A));
N_A_hat = zeros (run,length(N1_A));

sig = 0.001;

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

end %for get data

for m = 1:run

figure(m);
clf;

for b = 1:1:5

subplot(5,1,b);

rtime = zeros(5,2*length(N1_A) + 4*(b-1));
negtime = (-time*1000)/2:0.5:0;
postime = 1:0.5:(time*1000)/2 + 2*(b-1);
rtime(b,:) = horzcat(negtime,postime);
rtime(b,:) = rtime(b,:)./1000;

negtimenorm = (-time*1000)/2:0.5:0;
postimenorm = 1:0.5:(time*1000)/2;
rtimenorm = horzcat(negtimenorm,postimenorm);
rtimenorm = rtimenorm/1000;

shift = zeros(5,2*length(N1_A) + 4*(b-1));
ind = find(N_A(m,:));
gau = zeros(length(ind),time*2000);  
lam = zeros(1,length(ind));
Max = zeros(1,length(ind));
gwave = zeros(5,2*length(N1_A));

for j = 1:length(ind)
  
lam(j) = -time/2+0.001*ind(j);
gau(j,:) = normpdf(rtimenorm,lam(j),sig);
Max(j) = max(gau(j,:));
gau(j,:) = (gau(j,:)./Max(j))*N_A(m,ind(j));
gwave(b,:) = gwave(b,:) + gau(j,:);
shift(b,4*(b-1)+1:length(shift)) = gwave(b,:);

end %for j

plot(rtime(b,:),shift(b,:),'b');

hold on

%-------------------------------------------------------------------------Fit sine to and find phase of Gaussians---------------------------------------------------------------------- 

shift_nDC = zeros(5,2*length(N1_A)+4*(b-1));
shift_hat = zeros(5,2*length(N1_A)+4*(b-1));

shift_nDC(b,:) = shift(b,:) - mean(shift(b,:));
shift_hat(b,:) = fft(shift_nDC(b,:));

% Find the initial values for the sine fit

[imax,imaxindex] = max(abs(shift_hat(b,:)));

freq = (imaxindex-1)/time;

if freq > 120
   freq = freq/2;
elseif  freq < 40
   freq = freq*2;
end %if

B = 2*pi*freq;
phase = angle(shift_hat(b,imaxindex));

% Return to original data (graph starting at 0) to find amplitude and offset

shift_half = shift(b,1:length(shift)/2);
rtime_half = rtime(b,1:length(rtime)/2);
amp = (max(shift_half)-min(shift_half))/2;
offset = (max(shift_half)+min(shift_half))/2;


%Create a plot for each trial with sine fit for half of the points

%theta = [amp,B,phase,offset]; % start with correct values, since we know them
%control = {1000; 1; 1};
%[theta, obj_value, iterations, convergence] = bfgsmin("sinmodelnz", {theta, gwave_half, rtime_half}, control);
%z = theta(1)*cos(theta(2)*rtime_half+theta(3))+theta(4);

z = amp*cos(B*rtime_half+phase) + offset;

plot(rtime_half,z,'r');

hold on

totmax = max(N_A(m,:));
axis([-time/2 time/2+(4*b-1)/1000 0.1 totmax*1.5]);
ylabel('Rate (1/s)');
set(gca,'XTick',[]);

title(horzcat('Shift +',num2str(2*(b-1)),'ms'));

phase = num2str(phase);
phase_label = 'phase=';
phase_value = text(0-(time/6),totmax*1.3,strcat(phase_label,phase));
set(phase_value,'color',[0 0.6 0],'fontsize',20-2*run);

end % for shift

xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));
xlabel('Time (ms)');

end % for run

hold off
