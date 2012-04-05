function plotWeightsNew(simpleAFileName, simpleBFileName, simpleCFileName)

simpleA=readpvpfile(simpleAFileName);
simpleB=readpvpfile(simpleBFileName);
simpleC=readpvpfile(simpleCFileName);

[temp1 temp2 numarbors] = size(simpleA{1}.values);

[ysize xsize fsize]=size(simpleA{1}.values{1});
X=linspace(-floor(xsize/2),floor(xsize/2),xsize);
Y=linspace(-floor(ysize/2),floor(ysize/2),ysize);
T=linspace(0,-(numarbors-1),numarbors);

for(f=3) %[1:fsize])
    for(t=[1:numarbors])
        simpleW(t,:,:)=squeeze(simpleA{1}.values{t}(:,:,f))-squeeze(simpleB{1}.values{t}(:,:,f))-squeeze(simpleC{1}.values{t}(:,:,f));
    end
    
    if((f==2)||(f==4))
        weights=squeeze(simpleW(:,:,int32(xsize/2)));
    else
        weights=squeeze(simpleW(:,int32(ysize/2),:));
    end
        zmax=max(max(squeeze(simpleW(:,int32(ysize/2),:))))
        zmin=min(min(squeeze(simpleW(:,int32(ysize/2),:))))
    figure(10+f); surf(Y,X,squeeze(simpleW(3,:,:)));view(0,90);
    xlabel('X Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    ylabel('Y Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    set(gca, 'FontSize', 18, 'FontWeight', 'Bold');
    grid off
    axis tight
    title('T = -3', 'FontSize', 18, 'FontWeight', 'Bold');
    caxis([zmin zmax])
    figure(20+f); surf(Y,X,squeeze(simpleW(int32(numarbors/2),:,:)));view(0,90);
    xlabel('X Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    ylabel('Y Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    set(gca, 'FontSize', 18, 'FontWeight', 'Bold');
    grid off
    axis tight
    title('T = -5', 'FontSize', 18, 'FontWeight', 'Bold');
    caxis([zmin zmax])
    figure(30+f); surf(Y,X,squeeze(simpleW(8,:,:)));view(0,90);
    xlabel('X Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    ylabel('Y Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    set(gca, 'FontSize', 18, 'FontWeight', 'Bold');
    grid off
    axis tight
    title('T = -8', 'FontSize', 18, 'FontWeight', 'Bold');
    caxis([zmin zmax])
    %figure(40+f); surf(X,T,squeeze(simpleW(:,:,int32(ysize/2))));view(0,90);
    %figure(40+f); surf(X,T,weights);view(0,90);
    %figure(50+f); surf(Y,T,weights);view(0,90);
    figure(40+f); surf(X,T,squeeze(simpleW(:,:,int32(xsize/2))));view(0,90);
    caxis([zmin zmax])
    figure(50+f); surf(Y,T,squeeze(simpleW(:,int32(ysize/2),:)));view(0,90);
    xlabel('X Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    ylabel('Time (T) Axis', 'FontSize', 18, 'FontWeight', 'Bold');
    set(gca, 'FontSize', 18, 'FontWeight', 'Bold');
    grid off
    axis tight
    %title('T = -3', 'FontSize', 18, 'FontWeight', 'Bold');
    caxis([zmin zmax])
    %figure(50+f); plot(T,squeeze(simpleW(:,int32(xsize/2),int32(ysize/2))));
end