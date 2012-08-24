function plotHistOSI(allhist, c, osi, nori)
    h=allhist(:,c,:);
    for i=1:size(h,3)
        h(:,1,i)=h(:,1,i).*i*(360/nori);
    end
    figure;
    h=reshape(h,1,size(h,1)*size(h,2)*size(h,3));
    h(h==0) = [];
    h=h-(360/nori);
    hist(h, nori);
    xlim([-5 360]);
    xlabel('Orientation');
    ylabel('Count');
    title(['Cell ' num2str(c) '  OSI = ' num2str(osi)])
    box off;
end