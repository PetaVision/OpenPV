function TestCheckpoint(layer,deltatime)

close  all;

filename = ["../../gjkunde/graystart/Checkpoint1000/",layer,".pvp"]

checkV1000 = \
    readpvpfile(filename);

check_V1000 = checkV1000{1}.values;

figure;
imagesc(check_V1000);
mean(check_V1000(:))
std(check_V1000(:))

filename = ["../../gjkunde/amoebarun/Checkpoint",num2str(1000+deltatime),"/",layer,".pvp"]

checkV0 = \
    readpvpfile(filename);

check_V0 = checkV0{1}.values;


figure(2);
imagesc(check_V0);
mean(check_V0(:))
std(check_V0(:))