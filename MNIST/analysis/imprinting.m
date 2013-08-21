%%%%%%%%%%%% Initialization %%%%%%%%%%%

W{1}.values{1} = 2*rand([32 32 1 1400])-1;
L{1}.values{1} = 2*rand([1 1 10 1400])-1;
filename = "MNIST_0000.png";

%%%%%%%%%%% Calculations %%%%%%%%%%%%%%%
%% Note: you must be in one of the MNIST
%% directories to successfully run this
%% code. E.g. ~/Pictures/MNIST/0

for f = 0:9
	cd(["../" num2str(f)])
	j = randperm(1400);
	t = randperm(1000);
	label = zeros(1,1,10,1);
	label(:,:,f+1,:) = 1;
	for i = 1:150
		mim = imread([filename(1:end-4-length(num2str(t(i)))) num2str(t(i)) ".png"]);
		new = rand([32 32 1 1])-1;
		new(3:30,3:30,1,1) = new(3:30,3:30,1,1) + reshape(double(mim),[28 28 1 1])./255;
		W{1}.values{1}(:,:,:,j(i)) = new;
		new2 = rand([1 1 10 1]) - 1;
		L{1}.values{1}(:,:,:,j(i)) = new2 + label;
	end
end

%%%%%%% Optional Plotting for Sanity Check %%%%%%%%%

for i = 1:1400
	imagesc(squeeze(W{1}.values{1}(:,:,:,i))',[-1 1])
	colormap(gray)
	L{1}.values{1}(:,:,:,i)
	[~, tmptitle] = max(L{1}.values{1}(:,:,:,i),[],3);
	title(num2str(tmptitle))
	pause()
end

%%%%%%%%%%% Writing to PVP File %%%%%%%%%%%%%%

f = fopen("w_imprintedweights.pvp","r+b");
fseek(f,104,SEEK_SET);
for i = 1:1400
	fwrite(f,32,"int16");
	fwrite(f,32,"int16");
	fwrite(f,0,"int16");
	fwrite(f,0,"int16");
	tmp = squeeze(single(W{1}.values{1}(:,:,:,i)))';
	fwrite(f,tmp,"float32");
end
fclose(f);


f = fopen("w_imprintedlabels.pvp","r+b");
fseek(f,104,SEEK_SET);
for i = 1:1400
	fwrite(f,1,"int16");
	fwrite(f,1,"int16");
	fwrite(f,0,"int16");
	fwrite(f,0,"int16");
	tmp = single(L{1}.values{1}(:,:,:,i));
	fwrite(f,tmp,"float32");
end
fclose(f);