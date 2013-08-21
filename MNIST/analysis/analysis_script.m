%% Change to output directory, wherever that may be %%

output_dir = "/Users/bcrocker/Documents/workspace/HyPerHLCA2/output_testrun2/";
cd(output_dir)


%%%%%%%%%% Plotting Error Files %%%%%%%%%%%%%

%% Initialization %%
%% Change Params Here %%

labelstats = [pwd filesep "LabelError_Stats.txt"];
moviestats = [pwd filesep "MovieError_Stats.txt"];
v1stats = [pwd filesep "V1_Stats"];
statsdir = [pwd filesep "stats"];
beginidx = 1; %% make this end - however long you want the plot to be
display = 1000; %% the length of display period for an image

%% Calculations %%

plotstats(labelstats, "sigma", 0, statsdir, beginidx)
plotstats(moviestats, "sigma", 0, statsdir, beginidx)
plotstats(v1stats, "nnz", 1, statsdir, beginidx)
plotstats_skip(moviestats, "sigma", 0, [statsdir filesep "long"], display-1, display)

%%%%%%%%%% Plotting V1 Histograms %%%%%%%%%%%%

%% Initialization %%
%% Change Params Here %%

statsdir = [pwd filesep "stats"];
V1_file = [pwd filesep "a4_V1.pvp"];

%% Calculations %%

v1rank = v1plots(V1_file, statsdir)

%%%%%%%%%% Plotting Dictionary %%%%%%%%%%%%%%%

%% Initialization %%
%% Change Params Here %%

lastcheckpointndx = 28000000;
checkdir = [pwd filesep "Checkpoints" filesep "Checkpoint" num2str(lastcheckpointndx)];
labelweights = [checkdir filesep "V1ToLabelError_W.pvp"];
movieweights = [checkdir filesep "V1ToMovieError_W.pvp"];
statsdir = [pwd filesep "stats"];

%% Movie Dictionary %%
%% requires v1rank from v1plots

W = readpvpfile(movieweights);
W = W{end}.values{1};
L = readpvpfile(labelweights);
L = squeeze(L{end}.values);			
f = figure;
sz = get (0, "screensize");
set (gcf, "position", sz) 
for i = 1:50
	subplot(5,10,i)
	imagesc(squeeze(W(:,:,:,v1rank(i)))')
	[~, tmptitle] = max(squeeze(L(:,v1rank(i)))); 
	title(num2str(tmptitle-1))
	axis("off")
	colormap(gray)
end
saveas(f,[statsdir filesep "V1ToMovieWeights_bw_" num2str(lastcheckpointndx)],"png")

%% Label Dictionary %%

[~, maxnum] = max(L,[],2);
[maxnum,maxind] = sort(magnum);
g = figure;
imagesc(L(:,maxind))
saveas(g,[statsdir filesep "V1ToLabelWeights_bw_" num2str(lastcheckpointndx)],"png")

%% Plot the average movie weights for a label %%

label = 3; %% anything 1:nf
h = figure;
imagesc(squeeze(mean(W(:,:,1,maxind(maxnum==label))))')


%%%%%%%%%%%%% Checking Accuracy %%%%%%%%%%%%%%%

guessdir = "/Users/bcrocker/Documents/workspace/HyPerHLCA2/output_guessrun/";
cd(guessdir)
guess = readpvpfile("a1_labels.pvp");
recon = readpvpfile("a6_labelRecon.pvp");
for i = 1:100
 [~,g(i)] = max(guess{i+1}.values);
 [~,r(i)] = max(recon{i+1}.values);
end
accuracy = sum(g==r)





