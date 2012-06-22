run = 5



[time, Gin_A] = pvp_readPointLIFprobeRun(1,"GanglionON",{'A'});
[time, N1_A] = pvp_readPointLIFprobeRun(1,"N1GanglionON",{'A'});

N_A = zeros (run,8,length(N1_A)); % index is run,id,time

G_A = zeros (run,length(Gin_A));   % index is run,time
Gsum_A = zeros (length(Gin_A),1);   % index is run,time

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

Gsum_A = Gsum_A+Gin_A;

end %for

% Averaged spike rate by dividing by number of trials

Gsum_A = Gsum_A./run;

disp(Gsum_A);



rtime = 1:time;

clf;

LineLength = 0.1;



figure(1);

%Bar plot of sum of center Ganglion cell activity

top = subplot(3,1,1);

bar(rtime,Gsum_A,'facecolor','r','edgecolor','r');

axis([0 time 0 1.5*max(Gsum_A)])

title('Activity of Center Ganglion Cell and Neighbouring Ganglion Cells');
ylabel('Avg. Spike Rate (1/s)');

set(gca,'XTickLabel',[]);

set(top,'pos',[0.13,0.64,0.774,0.258]);

%Line plot of center Ganglion cell activity

middle = subplot(3,1,2);

for m=1:run

for i = 1:time

     if  G_A(m,i) ~= 0
      
       x = [i,i];

       y = [m*G_A(m,i)-LineLength,m*G_A(m,i)+LineLength];

       line(x,y,'color','r');
      
     end %if %center



end %for %i

end %for %m

axis([0 time 0.1 max(m)+1]);
set(gca,'YTick',[0:1]);
set(middle,'pos',[0.13,0.38,0.774,0.258]);

set(gca,'XTickLabel',[]);


ylabel('Trial #'); 

%Make line plot of Ganglion neighbours activity

bottom = subplot(3,1,3);

for m = 1:run

for j = 1:8

    for i = 1:time

       if N_A(m,j,i) ~= 0

         x = [i-0.4+0.1*i,i-0.4+0.1*i];

         y = [m*N_A(m,j,i)-LineLength,m*N_A(m,j,i)+LineLength];

         line(x,y,'color','b');

       end %if

     end %for

end %for


axis([0 time 0.1 max(m)+1]);
xlabel('Time (ms)');
ylabel('Trial #'); 

end %for

%pos = get(bottom,'pos');

%disp(pos);

%pos(2) = pos(2)+0.01;
set(gca,'YTick',[0:1]);
set(bottom,'pos',[0.13,0.1195,0.774,0.258]);

