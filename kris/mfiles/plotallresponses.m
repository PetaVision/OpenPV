close all
clear all
clc

simpath='/Users/kpeterson/Documents/workspace/kris/output/normalizedV1s/';

%plot simpleA1:
simpleA1File=[simpath, 'a3.pvp'];
simpleA1Mat=readpvpfile(simpleA1File);
impulserespfouriertransform(simpleA1Mat, 'SimpleA1', 100, 0, pi/2)


% simpleAFileName = [simpath, 'w2_last.pvp'];
% simpleBFileName = [simpath, 'w3_last.pvp'];
% simpleCFileName = [simpath, 'w4_last.pvp'];
% plotWeightsNew(simpleAFileName, simpleBFileName, simpleCFileName)

%plot simpleB1:
simpleB1File=[simpath, 'a4.pvp'];
simpleB1Mat=readpvpfile(simpleB1File);
impulserespfouriertransform(simpleB1Mat, 'SimpleB1', 200, 0, pi/2)


% simpleAFileName = [simpath, 'w2_last.pvp'];
% simpleBFileName = [simpath, 'w3_last.pvp'];
% simpleCFileName = [simpath, 'w4_last.pvp'];
% plotWeightsNew(simpleAFileName, simpleBFileName, simpleCFileName)

%plot simpleA2:
simpleA2File=[simpath, 'a8.pvp'];
simpleA2Mat=readpvpfile(simpleA2File);
impulserespfouriertransform(simpleA2Mat, 'SimpleA2', 300, -pi/4, -pi/2)


% simpleA2AFileName = [simpath, 'w14_last.pvp'];
% simpleA2BFileName = [simpath, 'w15_last.pvp'];
% simpleA2CFileName = [simpath, 'w16_last.pvp'];
% plotWeightsNew(simpleA2AFileName, simpleA2BFileName, simpleA2CFileName)

%plot simpleB2:
simpleB2File=[simpath, 'a9.pvp'];
simpleB2Mat=readpvpfile(simpleB2File);
impulserespfouriertransform(simpleB2Mat, 'SimpleB2', 400, -pi/4, -pi/2)

%plot simpleA3:
simpleA3File=[simpath, 'a13.pvp'];
simpleA3Mat=readpvpfile(simpleA3File);
impulserespfouriertransform(simpleA3Mat, 'SimpleA3', 500, 0, -pi/4)
% simpleA3AFileName = [simpath, 'w26_last.pvp'];
% simpleA3BFileName = [simpath, 'w27_last.pvp'];
% simpleA3CFileName = [simpath, 'w28_last.pvp'];
% plotWeightsNew(simpleA3AFileName, simpleA3BFileName, simpleA3CFileName)

%plot simpleB3:
simpleB3File=[simpath, 'a14.pvp'];
simpleB3Mat=readpvpfile(simpleB3File);
impulserespfouriertransform(simpleB3Mat, 'SimpleB3', 600, 0, -pi/4)

%plot simpleA4:
simpleA4File=[simpath, 'a18.pvp'];
simpleA4Mat=readpvpfile(simpleA4File);
impulserespfouriertransform(simpleA4Mat, 'SimpleA4', 700, 0, pi/2)
%plot simpleB4:
simpleB4File=[simpath, 'a19.pvp'];
simpleB4Mat=readpvpfile(simpleB4File);
impulserespfouriertransform(simpleB4Mat, 'SimpleB4', 800, 0, pi/2)