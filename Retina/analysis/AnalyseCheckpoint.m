ganglionON_V = \
    readpvpfile("../graystart/Checkpoint3/GanglionON_V.pvp");
size(ganglionON_V{1}.values)

colormap(gray);

 Vmem = ganglionON_V{1}.values;
