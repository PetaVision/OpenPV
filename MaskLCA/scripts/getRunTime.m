addpath("/home/wshainin/workspace/PetaVision/mlab/util/");
pvpFile  = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/a2_V1_S2.pvp";
FID      = fopen(pvpFile);
hdr      = readpvpheader(FID);
timeStep = hdr.nbands*hdr.time
