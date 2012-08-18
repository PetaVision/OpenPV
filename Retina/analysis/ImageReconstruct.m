tic

% Get Ganglion cell data
readpvpGanglionfile;

rtime = 1:numframes;

% Change 16438 rows into 128x128 matrix
N_A = full(N_A); %make non-sparse
dim = sqrt(length(N_A));
Gcells = zeros(dim,dim,numframes);

for rows = 1:dim
    for N_Asect = (rows-1)*dim+1:rows*dim
        Gcells(rows,N_Asect-(dim*(rows-1)),:) = N_A(N_Asect,:);
    end %for 128-long sections of N_A
end %for rows

%%---------------------------------------------------Image reconstruction using 'raw' Ganglion cells-----------------------------------------------

rawganglions = sum(Gcells,3);

figure(1);
clf;
rawimage = imagesc(rawganglions);
colormap(gray);

%%-------------------------------------------------------------Making wave for each cell neighbourhood---------------------------------------

% Get activity of 9 cell neighbourhood

center9 = zeros(dim*dim,numframes);

% Set corner values of center9

center9(1,:) = Gcells(1,2,:) + Gcells(2,2,:) + Gcells(2,1,:) + Gcells(1,1,:);
center9(dim,:) = Gcells(1,dim-1,:) + Gcells(2,dim-1,:) + Gcells(2,dim,:) + Gcells(1,dim,:);
center9(dim*dim-dim+1,:) = Gcells(dim,2,:) + Gcells(dim-1,2,:) + Gcells(dim-1,1,:) + Gcells(dim,1,:);
center9(dim*dim,:) = Gcells(dim,dim-1,:) + Gcells(dim-1,dim-1,:) + Gcells(dim-1,dim,:) + Gcells(dim,dim); 

% Set all other values of center9, taking edges into account


for i = 1:dim
    for j = 1:dim
        if (i == 1) && (j ~= 1) && (j ~= dim)
           center9((i-1)*dim+j,:) = Gcells(i,j,:)+Gcells(i,j+1,:)+Gcells(i+1,j+1,:)+Gcells(i+1,j,:)+Gcells(i+1,j-1,:)+Gcells(i,j-1,:);
        elseif (j == 1) && (i ~= 1) && (i ~=dim)
           center9((i-1)*dim+j,:) = Gcells(i,j,:)+Gcells(i,j+1,:)+Gcells(i+1,j+1,:)+Gcells(i+1,j,:)+Gcells(i-1,j,:)+Gcells(i-1,j+1,:);
        elseif (i == dim) && (j ~= 1) && (j ~= dim)
           center9((i-1)*dim+j,:) = Gcells(i,j,:)+Gcells(i,j+1,:)+Gcells(i,j-1,:)+Gcells(i-1,j-1,:)+Gcells(i-1,j,:)+Gcells(i-1,j+1,:);
        elseif (j == dim) && (i ~= 1) && (i ~= dim)
           center9((i-1)*dim+j,:) = Gcells(i,j,:)+Gcells(i+1,j,:)+Gcells(i+1,j-1,:)+Gcells(i,j-1,:)+Gcells(i-1,j-1,:)+Gcells(i-1,j,:);
        elseif (i ~= 1) && (i ~= dim) && (j ~= 1) && (j ~= dim) 
           center9((i-1)*dim+j,:) = Gcells(i,j,:)+Gcells(i,j+1,:)+Gcells(i+1,j+1,:)+Gcells(i+1,j,:)+Gcells(i+1,j-1,:)+...
           Gcells(i,j-1,:)+Gcells(i-1,j-1,:)+Gcells(i-1,j,:)+Gcells(i-1,j+1,:);
        end %if
    end %for cols
end %for rows

center9 = squeeze(center9);





% Determine and set initial vectors and variables

oscwave = zeros(dim*dim,numframes);
LGN_smart = zeros(dim,dim);

for cell = 1:dim*dim

pklocs = [];
range = [];

center9_nDC(cell,:) = center9(cell,:) - mean(center9(cell,:));
center9_hat(cell,:) = fft(center9_nDC(cell,:));

% Calculate frequency

[imax,imaxindex(cell)] = max(abs(center9_hat(cell,:)));
freq(cell) = ((imaxindex(cell)-1)*1000)/numframes;

% Calculate frequency bandwidth

if imaxindex(cell) < 6 || imaxindex(cell) > length(center9_hat(cell,:)) - 6
   FWHM(cell) = mean([FWHM(cell-1),FWHM(cell-2)]);
else FWHM(cell) = fwhm(abs(center9_hat(cell,imaxindex(cell)-5:imaxindex(cell)+5)));
end %if

if FWHM(cell) == 0
   FWHM(cell) = mean([FWHM(cell-1),FWHM(cell-2)]);
end %if
 
fb(cell) = ((1/(FWHM(cell)*2*pi))^2)*1000;
fc(cell) = freq(cell);

% Set support and grid parameters
lb = 1;
ub = numframes;
n = numframes;

cell

poscenter9 = find(center9(cell,:));
wav = zeros(length(poscenter9),numframes);

% Make CMWs and wave

  for j = 1:length(poscenter9)
      displ = -poscenter9(j);
      wav(j,:) = cmorwavf(lb,ub,n,fb(cell),fc(cell),displ);
  end %for j wav

oscwave(cell,:) = sum(abs(wav),1); 

%----------------------------------------------------------------Smart LGN cell---------------------------------------------------------

 % Finding peaks
        
    if oscwave(cell,1) > oscwave(cell,2) && (oscwave(cell,i) > max(oscwave(cell,:))*(7/10))
           pklocs(1) = 1;
    end %if %oscwave starts with peak
    for i = 2:length(oscwave(cell,:))-1
        if (oscwave(cell,i) > oscwave(cell,i-1)) && (oscwave(cell,i) > oscwave(cell,i+1)) && (oscwave(cell,i) > max(oscwave(cell,:))*(7/10))
           pklocs(length(pklocs)+1) = i;
        end %if peak, find index
    end %for indices of mwave
    if oscwave(cell,length(oscwave(cell,:))) > oscwave(cell,length(oscwave(cell,:))-1) && (oscwave(cell,i) > max(oscwave(cell,:))*(7/10))
           pklocs(length(pklocs)+1) = length(oscwave(cell,:));
    end %if %oscwave ends with peak (7/10 determined empirically)

    if isempty(pklocs)
           continue
    end %if all peaks of cell aren't above threshold

LGNrange = zeros(dim*dim,2,length(pklocs));

 % Finding average range of peaks (period) for computing FWHM
    pkrange = (numframes/length(poscenter9));
    pkrange = ceil(pkrange); %make integer for FWHM
    pkrange_half = ceil(pkrange/2);

 % Find full-width half maximum of peaks and set range in which center G cells cause response in smart LGN
    if pklocs(1) < 1 + pkrange/2
       LGNrange(cell,1,1) = rtime(1);
       LGNrange(cell,2,1) = rtime(pkrange_half);
       start = 2;
    else start = 1;
    end %if %oscwave starts with peak

    if pklocs(length(pklocs)) > length(rtime) - pkrange/2 - 1
       LGNrange(cell,1,length(pklocs)) = rtime(length(rtime)-pkrange_half); 
       LGNrange(cell,2,length(pklocs)) = rtime(length(rtime));
       last = length(pklocs)-1;       
    else last = length(pklocs);
    end %if %oscwave ends with peak

    for loc = start:last
        LGNrange(cell,1,loc) = rtime(pklocs(loc) - pkrange_half); % lower bound
        if pklocs(loc) + pkrange_half > length(rtime)
           LGNrange(cell,2,loc) = rtime(length(rtime));
        else LGNrange(cell,2,loc) = rtime(pklocs(loc) + pkrange_half); % upper bound
        end % if 2nd to last peak is near end of rtime/oscwave
    end %for locations of peaks

% Find at what times G cell is in range(make smart LGN cell)  

r = ceil(cell/dim);
c = rem(cell,dim);
    if c == 0
       c = dim;
    end %if

poscenter = find(Gcells(r,c,:));

for j = 1:length(poscenter)
    for l = 1:length(pklocs)
        if (rtime(poscenter(j)) > LGNrange(cell,1,l)) && (rtime(poscenter(j)) < LGNrange(cell,2,l))                             
           LGN_smart(r,c) = LGN_smart(r,c) + 1;
        end %if %make LGN smart
    end %for each peak
end %for positive values of cell 

end %for cell


%Plot properties of center cell of array (64,64)

figure(3);
clf;

% Plots CMW wave of neighbourhood
subplot(4,1,1);
plot(rtime,oscwave(8128,:))

% Plots ranges
subplot(4,1,2);
for l = 1:length(LGNrange(8128,1,:))
x1 = [LGNrange(8128,1,l),LGNrange(8128,1,l)];
x2 =  [LGNrange(8128,2,l),LGNrange(8128,2,l)];
y = [1,2];
line(x1,y);
line(x2,y);
end %for length(center cell range)

% Plots all spikes
subplot(4,1,3);

spikescenter = zeros(1,length(rtime));
poscenter = find(Gcells(64,64,:));

for j = 1:length(poscenter)
spikescenter(poscenter(j)) = 1;
end %for

plot(rtime,spikescenter);

% Plots spikes to the LGN
subplot(4,1,4);

posvals = zeros(1,length(rtime));

for j = 1:length(poscenter)
    for l = 1:length(pklocs)
        if (rtime(poscenter(j)) > LGNrange(cell,1,l)) && (rtime(poscenter(j)) < LGNrange(cell,2,l))                             
           posvals(poscenter(j)) = 1;
        end %if %make LGN smart
    end %for each peak
end %for positive values of cell 

plot(rtime,posvals);


%%---------------------------------------------------Image reconstruction using smart LGN cell-----------------------------------------------

figure(2);
clf;
LGNsmartimage = imagesc(LGN_smart);
colormap(gray);

toc
