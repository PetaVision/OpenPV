duration = 600;
[time,Ganglion_RU1] =PV_readPointLIFprobeA("GanglionRU1",duration);
[time,Ganglion_RD1] =PV_readPointLIFprobeA("GanglionRD1",duration);
[time,Ganglion_RU2] =PV_readPointLIFprobeA("GanglionRU2",duration);
[time,Ganglion_RD2] =PV_readPointLIFprobeA("GanglionRD2",duration);
[time,Ganglion_RU1l] =PV_readPointLIFprobeA("GanglionRU1l",duration);
[time,Ganglion_RD1l] =PV_readPointLIFprobeA("GanglionRD1l",duration);
[time,Ganglion_RU2l] =PV_readPointLIFprobeA("GanglionRU2l",duration);
[time,Ganglion_RD2l] =PV_readPointLIFprobeA("GanglionRD2l",duration);
[time,Ganglion_RU1r] =PV_readPointLIFprobeA("GanglionRU1r",duration);
[time,Ganglion_RD1r] =PV_readPointLIFprobeA("GanglionRD1r",duration);
[time,Ganglion_RU2r] =PV_readPointLIFprobeA("GanglionRU2r",duration);
[time,Ganglion_RD2r] =PV_readPointLIFprobeA("GanglionRD2r",duration);

gu1  = Ganglion_RU1(100:500) % only the stimulated time frame
gd1  = Ganglion_RD1(100:500) % only the stimulated time frame
gu2  = Ganglion_RU2(100:500) % only the stimulated time frame
gd2  = Ganglion_RD2(100:500) % only the stimulated time frame

gu1l  = Ganglion_RU1l(100:500) % only the stimulated time frame
gd1l  = Ganglion_RD1l(100:500) % only the stimulated time frame
gu2l  = Ganglion_RU2l(100:500) % only the stimulated time frame
gd2l  = Ganglion_RD2l(100:500) % only the stimulated time frame

gu1r  = Ganglion_RU1r(100:500) % only the stimulated time frame
gd1r  = Ganglion_RD1r(100:500) % only the stimulated time frame
gu2r  = Ganglion_RU2r(100:500) % only the stimulated time frame
gd2r  = Ganglion_RD2r(100:500) % only the stimulated time frame

clf;
xrange=50;
[R,lag] = xcorr(gu1,gu1,'unbiased'); lagt=lag',plot(lagt,R,"1"),xlim([-1.*xrange,xrange]),grid;hold on;
[R,lag] = xcorr(gu1,gu1l,'unbiased'); lagt=lag',plot(lagt,R,"2",xlim([-1.*xrange,xrange]));
[R,lag] = xcorr(gu1,gu1r,'unbiased'); lagt=lag',plot(lagt,R,"3"),xlim([-1.*xrange,xrange]);
hold off;