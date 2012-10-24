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
Y.lateralinhibition.integratedSpikeCount = zeros(colsizex,colsizey,nf,nt);

for k=1:nt
    t = timeindices(k);
    dirname = sprintf('Checkpoint%d/',t);
    Y.scaledinput(:,:,k) = readpvpfile1values([dirname 'scaled input_A.pvp']);
    Y.retinaon(:,:,k) = readpvpfile1values([dirname 'RetinaON_A.pvp']);
    Y.retinaoff(:,:,k) = readpvpfile1values([dirname 'RetinaOFF_A.pvp']);
    Y.lcalif.activity(:,:,:,k) = readpvpfile1values([dirname 'lcalif_A.pvp']);
    Y.lcalif.V(:,:,:,k) = readpvpfile1values([dirname 'lcalif_V.pvp']);
    Y.lcalif.Vth(:,:,:,k) = readpvpfile1values([dirname 'lcalif_Vth.pvp']);
    Y.lcalif.Vadpt(:,:,:,k) = readpvpfile1values([dirname 'lcalif_Vadpt.pvp']);
    Y.lcalif.G_E(:,:,:,k) = readpvpfile1values([dirname 'lcalif_G_E.pvp']);
    Y.lcalif.G_I(:,:,:,k) = readpvpfile1values([dirname 'lcalif_G_I.pvp']);
    Y.lcalif.G_IB(:,:,:,k) = readpvpfile1values([dirname 'lcalif_G_IB.pvp']);
    Y.lcalif.G_Gap(:,:,:,k) = readpvpfile1values([dirname 'lcalif_G_Gap.pvp']);
    Y.lcalif.integratedSpikeCount(:,:,:,k) = readpvpfile1values([dirname 'lcalif_integratedspikecount.pvp']);
    wtmp = readpvpfile1values([dirname 'RetinaONtoS1_W.pvp']);
    Y.ojakernelon.weights(:,:,:,k) = wtmp{1};
    Y.ojakernelon.inputFiringRate(:,:,k) = readpvpfile1values([dirname 'RetinaONtoS1_inputFiringRate.pvp']);
    Y.ojakernelon.outputFiringRate(:,:,:,k) = readpvpfile1values([dirname 'RetinaONtoS1_outputFiringRate.pvp']);
    wtmp = readpvpfile1values([dirname 'RetinaOFFtoS1_W.pvp']);
    Y.ojakerneloff.weights(:,:,:,k) = wtmp{1};
    Y.ojakerneloff.inputFiringRate(:,:,k) = readpvpfile1values([dirname 'RetinaOFFtoS1_inputFiringRate.pvp']);
    Y.ojakerneloff.outputFiringRate(:,:,:,k) = readpvpfile1values([dirname 'RetinaOFFtoS1_outputFiringRate.pvp']);
    wtmp = readpvpfile1values([dirname 'Lateral Inhibition_W.pvp']);
    Y.lateralinhibition.weights(:,:,:,:,k) = wtmp{1};
    Y.lateralinhibition.integratedSpikeCount(:,:,:,k) = readpvpfile1values([dirname 'Lateral Inhibition_integratedSpikeCount.pvp']);
    
    fprintf(1,'t=%d, index %d of %d\n', t, k, nt);
end

cd(curpwd);

function V = readpvpfile1values(dirname)
raw = readpvpfile(dirname);
V = raw{1}.values;