twoAFC = repmat(0.5, 4, 5);

%% 2FC
twoAFC(2,1) = 0.921;
twoAFC(3,1) = 0.992;
twoAFC(4,1) = 1.0;
twoAFC(5,1) = 1.0;

%% 4FC
twoAFC(2,2) = 0.892;
twoAFC(3,2) = 0.956;
twoAFC(4,2) = 0.981;
twoAFC(5,2) = 0.981;

%% 6FC
twoAFC(2,3) = 0.823;
twoAFC(3,3) = 0.878;
twoAFC(4,3) = 0.912;
twoAFC(5,3) = 0.914;

%% 8FC
twoAFC(2,4) = 0.752;
twoAFC(3,4) = 0.802;
twoAFC(4,4) = 0.820;
twoAFC(5,4) = 0.808;

figure;
lh = plot([0:4], twoAFC(:,1), "-r");
set(lh, "LineWidth", [2.0]);
box off
hold on
lh = plot([0:4], twoAFC(:,2), "-b");
set(lh, "LineWidth", [2.0]);
lh = plot([0:4], twoAFC(:,3), "-g");
set(lh, "LineWidth", [2.0]);
lh = plot([0:4], twoAFC(:,4), "-k");
set(lh, "LineWidth", [2.0]);
