function Phasediff(run)

figure(1);
clf;

% Making initial vectors

[time, Gin_A] = pvp_readPointLIFprobeRun(1,"GanglionON",{'A'});
[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2+1:1:0;
postime = 1:1:time/2;
rtime = horzcat(negtime,postime);
rtime = rtime./1000;
rtime_all = zeros(run,time);

negtimefine = -time/2:0.5:0;
postimefine = 1:0.5:time/2;
rtimefine = horzcat(negtimefine,postimefine);
rtimefine = rtimefine./1000;


N_A = zeros(run,9,length(N1_A));
Allsums_nDC =  zeros (run,length(N1_A));
Allsums_hat = zeros (run,length(N1_A));

freq = zeros(1,run);

LineLength = 0.4;


%%---------    reading in center and neighbour activities for "run" runs

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

N_A(m,1,:) = N1_A;
N_A(m,2,:) = N2_A;
N_A(m,3,:) = N3_A;
N_A(m,4,:) = N4_A;
N_A(m,5,:) = N5_A;
N_A(m,6,:) = N6_A;
N_A(m,7,:) = N7_A;
N_A(m,8,:) = N8_A;
N_A(m,9,:) = Gin_A;

SumN = N1_A + N2_A + N3_A + N4_A + N5_A + N6_A + N7_A + N8_A+Gin_A;
SumN = SumN';
Allsums(m,:) = SumN;

Allsums_nDC(m,:) = Allsums(m,:) - mean(Allsums(m,:));
Allsums_hat(m,:) = fft(Allsums_nDC(m,:));

ind = find(Allsums(m,:)); % Get length of phase

end %for

phase = zeros(run,length(ind));
time = time/1000;




for m = 1:run

[imax,imaxindex] = max(abs(Allsums_hat(m,:)));
freq(m) = (imaxindex-1)/time;

if freq(m) > 120
   freq(m) = freq(m)/2;
elseif  freq(m) < 40
   freq(m) = freq(m)*2;
end %if

%phase = angle(Allsums(m,imaxindex)); % If you want to use phase from Fourier, also you will need to make adjustments to line 130



% Find phase of Complex Morlet Wavelets


% Set bandwith and center frequency parameters

fb = 0.002;
fc = freq(m);

% Set support and grid parameters

lb = rtimefine(1)-50/1000;
ub = rtimefine(length(rtimefine))+50/1000;
n = 800;

ind = find(Allsums(m,:));
wav = zeros(length(ind),2*length(N1_A));

 for j = 1:length(ind)

    displ = (time/2-0.001*ind(j));

    wav(j,:) = cmorwavf(lb,ub,n,fb,fc,displ);

% Finding phase of each CMW across trials

    phase(m,j) = angle(wav(j,length(rtimefine)/2));

 end %for j wav 

end %for run

%--------------------------------------------------Plotting line blobs w/ phase adjustment-----------------------------

subplot(2,1,1);

for m = 1:run

 rtime_all(m,:) = rtime;

 %Phase shift

   for j = 1:length(ind)
      rtime_all(m,j) = rtime_all(m,j) - (phase(m,j))/(2*pi*freq(m));
   end %for j

for k = 1:5

    for i = 1:time*1000

       if N_A(m,k,i) ~= 0

         x = [rtime_all(m,i)-0.0004+0.00001*k,rtime_all(m,i)-0.0004+0.00002*k];

         y = [m*N_A(m,k,i)-LineLength,m*N_A(m,k,i)+LineLength];

         line(x,y,'color','b');

       end %if

     end %for %time

end %for %cells

end %for runs

title('Phase Adjusted');

set(gca,'XTickLabel',[]);
axis([-time/2-time/200 time/2+time/200 0.1 run*1.5]);
ylabel('Trial #');
set(gca,'YTick',[0:run]);

%------------------------------------------------Plotting line blobs w/out phase adjustment--------------------------------

subplot(2,1,2);

for m = 1:run

for k = 1:5

    for i = 1:time*1000

       if N_A(m,k,i) ~= 0

         x = [rtime(i)-0.0004+0.000005*k,rtime(i)-0.0004+0.000005*k];

         y = [m*N_A(m,k,i)-LineLength,m*N_A(m,k,i)+LineLength];

         line(x,y,'color','b');

       end %if

     end %for %time

end %for %cells

end %for runs

title('Not Phase Adjusted');

axis([-time/2-time/200 time/2+time/200 0.1 run*1.5]);
xlabel('Time (s)');
ylabel('Trial #');

set(gca,'YTick',[0:run]);
xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));
