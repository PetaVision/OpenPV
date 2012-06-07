function TestRun(layer,offset)

%close  all;

figure;

filename = ["../../gjkunde/graystart/Checkpoint",num2str(1000),"/",layer,".pvp"]
checkV = readpvpfile(filename);
check_V = checkV{1}.values;
subplot(6,2,1);
imagesc(check_V);
mean(check_V(:))
std(check_V(:))

for i=1:11

filename = ["../../gjkunde/amoebarun/Checkpoint",num2str(offset+(i-1)*100),"/",layer,".pvp"]
checkV = readpvpfile(filename);
check_V = checkV{1}.values;
subplot(6,2,i+1);

titlestring = [layer," @ ",num2str(offset+(i-1)*100)," msec"]
title(titlestring,"fontsize",15);
imagesc(check_V);
mean(check_V(:))
std(check_V(:))

end

