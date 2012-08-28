% Plots the 3 Kilian graphs of average spike rate, LGN spikes, and EPSPs (top, middle, bottom, respectively)

function Kilian(run)

%%---------------------------------------- MAKING INITIAL VECTORS AND GETTING DATA------------------------------------------------------------

[time, Gin_A] = pvp_readPointLIFprobeRun(1,"GanglionON",{'A'});
[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2+1:1:0;
postime = 1:1:time/2;
rtime = horzcat(negtime,postime);
rtime = rtime./1000;
rtime_all = zeros(run,time);
rtime_LGN = zeros(run,time);

negtimefine = -time/2+1:0.5:0;
postimefine = 1:0.5:time/2;
rtimefine = horzcat(-time/2+0.5,negtimefine,0.5,postimefine);
rtimefine = rtimefine./1000;

N_A = zeros (run,9,length(N1_A));
G_A = zeros (run,length(Gin_A));
Allsums = zeros (run,length(N1_A));
Allsums_nDC =  zeros (run,length(N1_A));
Allsums_hat = zeros (run,length(N1_A));
LGN = zeros (run,length(Gin_A));
LGN_nDC =  zeros (run,length(N1_A));
LGN_hat = zeros (run,length(N1_A));
SumT = zeros (1,length(N1_A));

freq = zeros(1,run);
phase_G = zeros(1,run);
phase_LGN = zeros(1,run);
FWHMg = zeros(1,run);
FWHMlgn = zeros(1,run);

sig = 0.001;

% Reading in center and neighbour activities for "run" runs
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

G_A(m,:)=Gin_A;
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
Allsums(m,:) = SumN;

%----------------------------------------------------Find phase of Complex Morlet Wavelets for all G cells-------------------------------


%Getting inital values for calculating phase for CMWs of sum of G cells

Allsums_nDC(m,:) = Allsums(m,:) - mean(Allsums(m,:));
Allsums_hat(m,:) = fft(Allsums_nDC(m,:));

%ind = find(Allsums(m,:)); % Get length of phase for CMWs

SumT = SumT + Allsums(m,:);

end %for

time = time/1000;

% Determine frequency bandwidth

for m = 1:run
[imax,imaxindex] = max(abs(Allsums_hat(m,:)));
FWHMg(m) = fwhm(abs(Allsums_hat(m,imaxindex-5:imaxindex+5)));
if FWHMg(m) == 0
   error(strcat('Trial',num2str(m),'full-width half maximum of ganglions equals zero'));
end %if

freq(m) = (imaxindex-1)/time;

if freq(m) > 120
   freq(m) = freq(m)/2;
elseif  freq(m) < 40
   freq(m) = freq(m)*2;
end %if

%phase(m) = angle(max(Allsums(m,:))); % If you want to use phase from Fourier

ind = find(Allsums(m,1:length(Allsums(m,:))/2));

% Set bandwith and center frequency parameters

fb = ((1/(2*pi*FWHMg(m)))^2)/50;
fc = freq(m);

% Set support and grid parameters

lb = rtimefine(1);
ub = rtimefine(length(rtimefine));
n = time*2000;

wav = zeros(length(ind),2*length(N1_A));

 for j = 1:length(ind)

    displ = (time/2-0.001*ind(j));
    wav(j,:) = cmorwavf(lb,ub,n,fb,fc,displ);

  end %for j wav

% Superimpose CMWs

mwave(m,:) = sum(wav,1);

figure(6);

plot(rtimefine,real(mwave(m,:)))

%plot(rtimefine,wav(length(ind),:));

 rtime_all(m,:) = rtime;
 rtimefine_all(m,:) = rtimefine;

% Phase shift

figure(3);
clf;
plot(rtimefine,wav(length(ind),:));

ind_long = find(Allsums(m,:));

phase_G(m) = angle(mwave(m,length(rtimefine)/2))

   for j = 1:length(ind_long)
      rtime_all(m,ind_long(j)) = rtime_all(m,ind_long(j)) + phase_G(m)/(2*pi*freq(m));
   end %for j

rtimefine_all(m,:) = rtimefine_all(m,:) + phase_G(m)/(2*pi*freq(m));

end %for %run



% Find LGN cell data

for m = 1:run

    for k = 1:8 

        for i = 1:time*1000
                         
              if (N_A(m,k,i)~=0) && (G_A(m,i)~=0)

                 LGN(m,i) = 1;
           
              end %if

        end %for indices
    
    end %for neighbours

end %for run

%----------------------------------------------------Find phase of Complex Morlet Wavelets for LGN cell-------------------------------


%Getting inital values for calculating phase for CMWs of LGN cell

for m = 1:run

LGN_nDC(m,:) = LGN(m,:) - mean(LGN(m,:));
LGN_hat(m,:) = fft(LGN_nDC(m,:));

[imax,imaxindex] = max(abs(LGN_hat(m,1:length(LGN_hat(m,:))/2)));
freq(m) = (imaxindex-1)/time;

if freq(m) > 120
   freq(m) = freq(m)/2;
elseif  freq(m) < 40
   freq(m) = freq(m)*2;
end %if

%phase(m) = angle(max(LGN_hat(m,:))); % If you want to use phase from Fourier

% Calculate frequency bandwidth

FWHMlgn(m) = fwhm(abs(LGN_hat(m,imaxindex-4:imaxindex+4)));

if FWHMlgn(m) == 0
   error(strcat('Trial',num2str(m),'full-width half maximum of LGN equals zero'));
end %if

ind = find(LGN(m,1:length(LGN(m,:))/2));

% Set bandwith and center frequency parameters

fb = ((1/(FWHMlgn(m)*2*pi))^2)/50;
fc = freq(m);

% Set support and grid parameters

lb = rtimefine(1);
ub = rtimefine(length(rtimefine));
n = time*2000;

wav = zeros(length(ind),2*length(N1_A));

 for j = 1:length(ind)

    displ = (time/2-0.001*ind(j));
    wav(j,:) = cmorwavf(lb,ub,n,fb,fc,displ);

% Finding phase of each CMW across trials

%    phase_LGNs(m,j) = angle(wav(j,length(rtimefine)/2));

  end %for j wav

% Superimpose CMWs

LGNwave(m,:) = sum(wav,1);

% Phase shift

figure(4);
clf;
plot(rtimefine,wav(length(ind),:));

 rtime_LGN(m,:) = rtime;
 rtimefine_LGN(m,:) = rtimefine;

ind_long = find(LGN(m,:));
phase_LGN(m) = angle(LGNwave(m,length(rtimefine)/2));

% Phase shift

   for j = 1:length(ind_long)
      rtime_LGN(m,ind_long(j)) = rtime_LGN(m,ind_long(j)) + phase_LGN(m)/(2*pi*freq(m));
   end %for j

 rtimefine_LGN(m,:) = rtimefine_LGN(m,:) + phase_LGN(m)/(2*pi*freq(m));

end %for %run


% Line params

linelength = 0.4;
linewidth = 0.3;

%%--------------------------------------------------------NON-ADJUSTED PLOTTING------------------------------------------------------------


figure(1);
clf;


%-----------------------------------------------------Averaged Spike rate histogram-------------------------------------------------------

% Averaged spike rate by dividing by number of trials

SumT = SumT./run;

top = subplot(3,1,1);

%bar(rtime,SumT,'facecolor','r','edgecolor','r');
rtimep5 = rtime - 0.0005;
plot(rtimep5,SumT,'r','LineWidth',1.5);

axis([-time/2-time/50 time/2+time/50 0 3*max(SumT)])
title('Non-Phase Adjusted Activity of LGN Cell and Neighbouring Ganglion Cells');
ylabel('Avg. Spike Rate (1/s)');

set(gca,'XTickLabel',[]);

set(top,'pos',[0.13,0.64,0.774,0.258]);

%------------------------------------------------Line plot of LGN cell activity--------------------------------------------------

middle = subplot(3,1,2);

for m = 1:run

        for i = 1:time*1000
                         
              if LGN(m,i) ~= 0

                 x = [rtime(i),rtime(i)];

                 y = [m-linelength,m+linelength];

                 line(x,y,'color','r','LineWidth',linewidth+0.1);
      
              end %if

        end %for %i       

hold on

plot(rtimefine,(1/3).*(real(LGNwave(m,:))./max(real(LGNwave(m,:))))+m)

end %for %m

axis([-time/2-time/50 time/2+time/50 0.1 max(m)*1.5]);
set(gca,'YTick',[0:max(m)]);
set(middle,'pos',[0.13,0.38,0.774,0.258]);

set(gca,'XTickLabel',[]);


ylabel('Trial #'); 

%------------------------------------------------Line plot of all Ganglion cells activity-----------------------------------------------------

bottom = subplot(3,1,3);

for m = 1:run

for k = 1:9

    for i = 1:time*1000

       if N_A(m,k,i) ~= 0

         x = [rtime(i)-0.0004+0.0001*k,rtime(i)-0.0004+0.0001*k];

         y = [m*N_A(m,k,i)-linelength,m*N_A(m,k,i)+linelength];

         line(x,y,'color','b','LineWidth',linewidth);

       end %if

     end %for %time

hold on

plot(rtimefine,(1/3).*(real(mwave(m,:))./max(real(mwave(m,:))))+m)

end %for %cells



axis([-time/2-time/50 time/2+time/50 0.1 run*1.5]);
xlabel('Time (s)');
ylabel('Trial #');

end %for %runs

set(gca,'YTick',[0:run]);
set(bottom,'pos',[0.13,0.11958,0.774,0.258]);

xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));


%%----------------------------------------------------------ADJUSTED PLOTTING----------------------------------------------------------------

figure(2);
clf;

%-----------------------------------------------------Adjusted Average Spike Rate----------------------------------------------------------

% Make vector of rtime values from all phase shifts

rtime_diff = rtime;

for m = 1:run
    for i = 1:time*1000
        if rtime_all(m,i) ~= rtime(i)
           rtime_diff(length(rtime_diff)+1) = rtime_all(m,i);
        end %if
    end %for indices
end %for runs

SumT_diff = zeros(1,length(rtime_diff));

% Sum over trials

% If you want individual cells to affect height

for m = 1:run
    for ib = 1:length(rtime_diff)
        for i = 1:length(rtime)
            if rtime_diff(ib) == rtime_all(m,i);
               SumT_diff(ib) = SumT_diff(ib) + Allsums(m,i);
            end % if
        end %for rtime_all
    end %for rtime_diff
end %for runs

% Find histogram values and adjust them for num. cells spiking

shift_count = histc(rtime_diff(length(rtime)+1:length(rtime_diff)),rtime);


[rtime_shift,locs] = sort(rtime_diff);
for ix = 1:length(locs)
SumT_shift(ix) = SumT_diff(locs(ix));
end %for sort rtime values accordingly

indhist = find(shift_count);
indSumT_shift = find(SumT_shift);
start = 1;

for j = 1:length(indhist)
%    if j == 42
%       keyboard
%    end %if
	  heights(j) = sum(SumT_shift(indSumT_shift(start:start-1+shift_count(indhist(j)))));
    start = start + shift_count(indhist(j));
    shift_count(indhist(j)) = (shift_count(indhist(j))/shift_count(indhist(j)))*heights(j);
end %for nz shift_count

shift_count = shift_count./run;


%-----------------------------------------------------Averaged Spike Rate Histogram-------------------------------------------------------

% Averaged spike rate by dividing by number of trials

top = subplot(3,1,1);
plot(rtimep5,shift_count,'r','LineWidth',1.5);

%plot(rtime_shift,SumT_shift,'r','LineWidth',1.5);

axis([-time/2-time/50 time/2+time/50 0 3*max(SumT)])

title('Phase Adjusted Activity of LGN Cell and Neighbouring Ganglion Cells');
ylabel('Avg. Spike Rate (1/s)');

set(gca,'XTickLabel',[]);

set(top,'pos',[0.13,0.64,0.774,0.258]);

%------------------------------------------------Line plot of LGN cell activity--------------------------------------------------

middle = subplot(3,1,2);

for m = 1:run

        for i = 1:time*1000
                         
              if LGN(m,i) ~= 0

                 x = [rtime_LGN(m,i),rtime_LGN(m,i)];

                 y = [m-linelength,m+linelength];

                 line(x,y,'color','r','LineWidth',linewidth+0.1);
      
              end %if

        end %for %i       

hold on

plot(sort(rtimefine_LGN(m,:)),(1/3).*(real(LGNwave(m,:))./max(real(LGNwave(m,:))))+m)

end %for %m

axis([-time/2-time/50 time/2+time/50 0.1 max(m)*1.5]);
set(gca,'YTick',[0:max(m)]);
set(middle,'pos',[0.13,0.38,0.774,0.258]);

set(gca,'XTickLabel',[]);


ylabel('Trial #'); 

%--------------------------------------------Line plot of all Ganglion cells activity------------------------------------------------------

bottom = subplot(3,1,3);

for m = 1:run

for k = 1:9

    for i = 1:time*1000

       if N_A(m,k,i) ~= 0

         x = [rtime_all(m,i)-0.0004+0.0001*k,rtime_all(m,i)-0.0004+0.0001*k];

         y = [m*N_A(m,k,i)-linelength,m*N_A(m,k,i)+linelength];

         line(x,y,'color','b','LineWidth',linewidth);

       end %if

     end %for %time

end %for %cells

hold on

plot(sort(rtimefine_all(m,:)),(1/3).*(real(mwave(m,:))./max(real(mwave(m,:))))+m)

axis([-time/2-time/50 time/2+time/50 0.1 run*1.5]);
xlabel('Time (s)');
ylabel('Trial #');

end %for %runs

set(gca,'YTick',[0:run]);
set(bottom,'pos',[0.13,0.11958,0.774,0.258]);

xneg = -time/2:0.1:0;
xpos = 0:0.1:time/2;
set(gca,'XTick',horzcat(xneg,xpos));




%------------------------------------------------------------------Information rate: Phase-shifted vs. non-phase-shifted------------------------------------------


SumT_shift = shift_count;
spikerate = zeros(1,length(SumT));

spikerate = SumT./time;
spikeratenz = find(spikerate);
avgrate = mean(spikerate(spikeratenz));
spikerate_shift = SumT_shift./time;
spikerate_shiftnz = find(spikerate_shift);
avgrate_shift = mean(spikerate_shift(spikerate_shiftnz));

inform = sum((spikerate(spikeratenz)./avgrate).*log2(spikerate(spikeratenz)./avgrate));
inform_shift = sum((spikerate_shift(spikerate_shiftnz)./avgrate).*log2(spikerate_shift(spikerate_shiftnz)./avgrate));

informdiff = inform_shift-inform;

disp('Information Increase =');
disp(informdiff);

keyboard