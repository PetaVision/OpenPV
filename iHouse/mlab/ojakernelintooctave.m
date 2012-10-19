function Y = ojakernelintooctave(cpdirname, timeindices)
% Y = ojakernelintooctave(timeindices)
% Y is a struct with a whole bunch of fields

curpwd = pwd;
cd(cpdirname);

nt = numel(timeindices);

colsizex = 32;
colsizey = 32;
nf = 16;
feedforwardweightx = 5;
feedforwardweighty = 5;
lateralweightx = 9;
lateralweighty = 9;
lcalifmargin = 8;

Y.scaledinput = zeros(colsizex,colsizey,nt);
Y.retinaon = zeros(colsizex,colsizey,nt);
Y.retinaoff = zeros(colsizex,colsizey,nt);
Y.lcalif.activity = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.V = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.Vth = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.Vadpt = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.G_E = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.G_I = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.G_IB = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.G_Gap = zeros(colsizex,colsizey,nf,nt);
Y.lcalif.integratedSpikeCount = zeros(colsizex,colsizey,nf,nt);
Y.ojakernelon.weights = zeros(feedforwardweightx,feedforwardweighty,nf,nt);
Y.ojakernelon.inputFiringRate = zeros(colsizex,colsizey,nt);
Y.ojakernelon.outputFiringRate = zeros(colsizex,colsizey,nf,nt);
Y.ojakerneloff.weights = zeros(feedforwardweightx,feedforwardweighty,nf,nt);
Y.ojakerneloff.inputFiringRate = zeros(colsizex,colsizey,nt);
Y.ojakerneloff.outputFiringRate = zeros(colsizex,colsizey,nf,nt);
Y.lateralinhibition.weights = zeros(lateralweightx,lateralweighty,nf,nf,nt);
Y.lateralinhibition.integratedSpikeCount = zeros(64,64,nf,nt); % zeros(colsizex+2*lcalifmargin,colsizey+2*lcalifmargin,nf,nt);

for k=1:nt
    t = timeindices(k);
    dirname = sprintf('Checkpoint%d/',t);
    Y.scaledinput(:,:,k) = readpvpfile([dirname 'scaled input_A.pvp']){1}.values;
    Y.retinaon(:,:,k) = readpvpfile([dirname 'RetinaON_A.pvp']){1}.values;
    Y.retinaoff(:,:,k) = readpvpfile([dirname 'RetinaOFF_A.pvp']){1}.values;
    Y.lcalif.activity(:,:,:,k) = readpvpfile([dirname 'lcalif_A.pvp']){1}.values;
    Y.lcalif.V(:,:,:,k) = readpvpfile([dirname 'lcalif_V.pvp']){1}.values;
    Y.lcalif.Vth(:,:,:,k) = readpvpfile([dirname 'lcalif_Vth.pvp']){1}.values;
    Y.lcalif.Vadpt(:,:,:,k) = readpvpfile([dirname 'lcalif_Vadpt.pvp']){1}.values;
    Y.lcalif.G_E(:,:,:,k) = readpvpfile([dirname 'lcalif_G_E.pvp']){1}.values;
    Y.lcalif.G_I(:,:,:,k) = readpvpfile([dirname 'lcalif_G_I.pvp']){1}.values;
    Y.lcalif.G_IB(:,:,:,k) = readpvpfile([dirname 'lcalif_G_IB.pvp']){1}.values;
    Y.lcalif.G_Gap(:,:,:,k) = readpvpfile([dirname 'lcalif_G_Gap.pvp']){1}.values;
    Y.lcalif.integratedSpikeCount(:,:,:,k) = readpvpfile([dirname 'lcalif_integratedspikecount.pvp']){1}.values;
    Y.ojakernelon.weights(:,:,:,k) = readpvpfile([dirname 'RetinaONtoS1_W.pvp']){1}.values{1};
    Y.ojakernelon.inputFiringRate(:,:,k) = readpvpfile([dirname 'RetinaONtoS1_inputFiringRate.pvp']){1}.values;
    Y.ojakernelon.outputFiringRate(:,:,:,k) = readpvpfile([dirname 'RetinaONtoS1_outputFiringRate.pvp']){1}.values;
    Y.ojakerneloff.weights(:,:,:,k) = readpvpfile([dirname 'RetinaOFFtoS1_W.pvp']){1}.values{1};
    Y.ojakerneloff.inputFiringRate(:,:,k) = readpvpfile([dirname 'RetinaOFFtoS1_inputFiringRate.pvp']){1}.values;
    Y.ojakerneloff.outputFiringRate(:,:,:,k) = readpvpfile([dirname 'RetinaOFFtoS1_outputFiringRate.pvp']){1}.values;
    Y.lateralinhibition.weights(:,:,:,:,k) = readpvpfile([dirname 'Lateral Inhibition_W.pvp']){1}.values{1};
    Y.lateralinhibition.integratedSpikeCount(:,:,:,k) = readpvpfile([dirname 'Lateral Inhibition_integratedSpikeCount.pvp']){1}.values;
    
    fprintf(1,'t=%d, index %d of %d\n', t, k, nt);
    fflush(1);
end

cd(curpwd);