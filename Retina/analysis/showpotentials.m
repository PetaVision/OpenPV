figure
colormap("gray")
PAAON_V5 = readpvpfile("../graystart/Checkpoint5/PAAmacrineON_V.pvp");
PAA = PAAON_V5{1}.values;
imagesc(PAA)
colorbar
mean(PAA(:))
std(PAA(:))
figure
colormap("gray")
GanglionON_V5 = readpvpfile("../graystart/Checkpoint5/GanglionON_V.pvp");
Ganglion = GanglionON_V5{1}.values;
imagesc(Ganglion)
colorbar
mean(Ganglion(:))
std(Ganglion(:))
