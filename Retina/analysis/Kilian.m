% Plots the 3 Kilian graphs of average spike rate, LGN spikes, and EPSPs (top, middle, bottom, respectively)

function Kilian(run)

% Making initial vectors

[time, Gin_A] = pvp_readPointLIFprobeRun(1,"GanglionON",{'A'});
[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

negtime = -time/2+1:1:0;
postime = 1:1:time/2;
rtime = horzcat(negtime,postime);
rtime = rtime./1000;
rtime_all = zeros(run,time);

N_A = zeros (run,9,length(N1_A));
G_A = zeros (run,length(Gin_A));
Allsums = zeros (run,length(N1_A));
SumT = zeros (1,length(N1_A));


phase = zeros(1,run);
freq = zeros(1,run);


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
Allsums_nDC =  zeros (run,length(N1_A));
Allsums_hat = zeros (run,length(N1_A));


%Calculating phase of each trial

Allsums_nDC(m,:) = Allsums(m,:) - mean(Allsums(m,:));
Allsums_hat(m,:) = fft(Allsums_nDC(m,:));
phase(m) = angle(max(Allsums(m,:)));

[imax,imaxindex] = max(abs(Allsums_hat(m,:)));
freq(m) = (imaxindex-1)/time;

if freq(m) > 120
   freq(m) = freq(m)/2;
elseif  freq(m) < 40
   freq(m) = freq(m)*2;
end %if


SumT = SumT + Allsums(m,:);

end %for

time_sec = time/1000;

% Averaged spike rate by dividing by number of trials

SumT = SumT./run;

LineLength = 0.2;

clf;

figure(1);

%Bar plot of sum of center Ganglion cell activity

top = subplot(3,1,1);

bar(rtime,SumT,'facecolor','r','edgecolor','r');

axis([-time_sec/2 time_sec/2 0 1.5*max(SumT)])

title('Activity of LGN Cell and Neighbouring Ganglion Cells');
ylabel('Avg. Spike Rate (1/s)');

set(gca,'XTickLabel',[]);

set(top,'pos',[0.13,0.64,0.774,0.258]);

%Line plot of center Ganglion cell activity

middle = subplot(3,1,2);

for m = 1:run

    for j = 1:8 

        for i = 1:time
                         
              if (N_A(m,j,i)~=0) && (G_A(m,i)~=0)

                 x = [rtime(i),rtime(i)];

                 y = [m-LineLength,m+LineLength];

                 line(x,y,'color','r');
      
              end %if

        end %for %i

    end %for j        

end %for %m

axis([-time_sec/2 time_sec/2 0.1 max(m)*1.5]);
set(gca,'YTick',[0:max(m)]);
set(middle,'pos',[0.13,0.38,0.774,0.258]);

set(gca,'XTickLabel',[]);


ylabel('Trial #'); 

%Make line plot of Ganglion neighbours activity

bottom = subplot(3,1,3);

for m = 1:run

rtime_all(m,:) = rtime;

rtime_all(m,:) = rtime_all(m,:);

%- phase(m)/(1000*2*pi*freq(m));

for j = 1:9

    for i = 1:time

       if N_A(m,j,i) ~= 0

         x = [rtime_all(m,i)-0.0004+0.00002*j,rtime_all(m,i)-0.0004+0.00002*j];

         y = [m*N_A(m,j,i)-LineLength,m*N_A(m,j,i)+LineLength];

         line(x,y,'color','b');

       end %if

     end %for %time

end %for %cells

axis([-time_sec/2 time_sec/2 0.1 run*1.5]);
xlabel('Time (s)');
ylabel('Trial #');

end %for %runs

%pos = get(bottom,'pos');

%disp(pos);

%pos(2) = pos(2)+0.01;
set(gca,'YTick',[0:run]);
set(bottom,'pos',[0.13,0.11958,0.774,0.258]);

xneg = -time_sec/2:0.1:0;
xpos = 0:0.1:time_sec/2;
set(gca,'XTick',horzcat(xneg,xpos));
