function TestRunCompare(layer,k,cor)

%close  all;

figure;
filename = ["../../gjkunde/graystart/Checkpoint",num2str(1000),"/",layer,".pvp"]
checkVr = readpvpfile(filename);
check_Vr = checkVr{1}.values;
subplot(3,1,1)
imagesc(check_Vr);
mean(check_Vr(:));
std(check_Vr(:));


filename = ["../../gjkunde/amoebarun/Checkpoint",num2str(1000),"/",layer,".pvp"]
checkV = readpvpfile(filename);
check_V = checkV{1}.values;
subplot(3,1,2);
imagesc(check_V);
mean(check_V(:));
std(check_V(:));

size(check_V)
check_V(k+1)
check_V(cor+1,cor+1)


change = check_V - check_Vr;
subplot(3,1,3);
imagesc(change);
mean(change(:));
std(change(:));
