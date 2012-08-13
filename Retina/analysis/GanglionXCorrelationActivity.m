
[time,GanglionON_RU1] =pvp_readPointLIFprobe("GanglionONRU1",{'A'});
[time,GanglionON_RD1] =pvp_readPointLIFprobe("GanglionONRD1",{'A'});
[time,GanglionON_RU2] =pvp_readPointLIFprobe("GanglionONRU2",{'A'});
[time,GanglionON_RD2] =pvp_readPointLIFprobe("GanglionONRD2",{'A'});
[time,GanglionON_RU1l] =pvp_readPointLIFprobe("GanglionONRU1l",{'A'});
[time,GanglionON_RD1l] =pvp_readPointLIFprobe("GanglionONRD1l",{'A'});
[time,GanglionON_RU2l] =pvp_readPointLIFprobe("GanglionONRU2l",{'A'});
[time,GanglionON_RD2l] =pvp_readPointLIFprobe("GanglionONRD2l",{'A'});
[time,GanglionON_RU1r] =pvp_readPointLIFprobe("GanglionONRU1r",{'A'});
[time,GanglionON_RD1r] =pvp_readPointLIFprobe("GanglionONRD1r",{'A'});
[time,GanglionON_RU2r] =pvp_readPointLIFprobe("GanglionONRU2r",{'A'});
[time,GanglionON_RD2r] =pvp_readPointLIFprobe("GanglionONRD2r",{'A'});

[time,GanglionOFF_RU1] =pvp_readPointLIFprobe("GanglionOFFRU1",{'A'});
[time,GanglionOFF_RD1] =pvp_readPointLIFprobe("GanglionOFFRD1",{'A'});
[time,GanglionOFF_RU2] =pvp_readPointLIFprobe("GanglionOFFRU2",{'A'});
[time,GanglionOFF_RD2] =pvp_readPointLIFprobe("GanglionOFFRD2",{'A'});
[time,GanglionOFF_RU1l] =pvp_readPointLIFprobe("GanglionOFFRU1l",{'A'});
[time,GanglionOFF_RD1l] =pvp_readPointLIFprobe("GanglionOFFRD1l",{'A'});
[time,GanglionOFF_RU2l] =pvp_readPointLIFprobe("GanglionOFFRU2l",{'A'});
[time,GanglionOFF_RD2l] =pvp_readPointLIFprobe("GanglionOFFRD2l",{'A'});
[time,GanglionOFF_RU1r] =pvp_readPointLIFprobe("GanglionOFFRU1r",{'A'});
[time,GanglionOFF_RD1r] =pvp_readPointLIFprobe("GanglionOFFRD1r",{'A'});
[time,GanglionOFF_RU2r] =pvp_readPointLIFprobe("GanglionOFFRU2r",{'A'});
[time,GanglionOFF_RD2r] =pvp_readPointLIFprobe("GanglionOFFRD2r",{'A'});

%ON

guON1  = GanglionON_RU1(100:500) % only the stimulated time frame
gdON1  = GanglionON_RD1(100:500) % only the stimulated time frame
guON2  = GanglionON_RU2(100:500) % only the stimulated time frame
gdON2  = GanglionON_RD2(100:500) % only the stimulated time frame

guON1l  = GanglionON_RU1l(100:500) % only the stimulated time frame
gdON1l  = GanglionON_RD1l(100:500) % only the stimulated time frame
guON2l  = GanglionON_RU2l(100:500) % only the stimulated time frame
gdON2l  = GanglionON_RD2l(100:500) % only the stimulated time frame

guON1r  = GanglionON_RU1r(100:500) % only the stimulated time frame
gdON1r  = GanglionON_RD1r(100:500) % only the stimulated time frame
guON2r  = GanglionON_RU2r(100:500) % only the stimulated time frame
gdON2r  = GanglionON_RD2r(100:500) % only the stimulated time frame

%OFF

guOFF1  = GanglionOFF_RU1(100:500) % only the stimulated time frame
gdOFF1  = GanglionOFF_RD1(100:500) % only the stimulated time frame
guOFF2  = GanglionOFF_RU2(100:500) % only the stimulated time frame
gdOFF2  = GanglionOFF_RD2(100:500) % only the stimulated time frame

guOFF1l  = GanglionOFF_RU1l(100:500) % only the stimulated time frame
gdOFF1l  = GanglionOFF_RD1l(100:500) % only the stimulated time frame
guOFF2l  = GanglionOFF_RU2l(100:500) % only the stimulated time frame
gdOFF2l  = GanglionOFF_RD2l(100:500) % only the stimulated time frame

guOFF1r  = GanglionOFF_RU1r(100:500) % only the stimulated time frame
gdOFF1r  = GanglionOFF_RD1r(100:500) % only the stimulated time frame
guOFF2r  = GanglionOFF_RU2r(100:500) % only the stimulated time frame
gdOFF2r  = GanglionOFF_RD2r(100:500) % only the stimulated time frame


clf;
xrange=50;
[R,lag] = xcorr(guON1,guOFF1,'unbiased'); lagt=lag',plot(lagt,R,"1"),xlim([-1.*xrange,xrange]),grid;hold on;
[R,lag] = xcorr(guON1,guON1l,'unbiased'); lagt=lag',plot(lagt,R,"2",xlim([-1.*xrange,xrange]));
[R,lag] = xcorr(guON1,guON1r,'unbiased'); lagt=lag',plot(lagt,R,"3"),xlim([-1.*xrange,xrange]);
hold off;
