close all;
clear all;
global filename;
global output_path;  output_path  = '/Users/slundquist/Documents/workspace/BIDS/output/';

norm_NLI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_norm_NLI.pvp';
norm_LI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_norm_LI.pvp';
strong_NLI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_strong_NLI.pvp';
strong_LI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_strong_LI.pvp';
weak_NLI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_weak_NLI.pvp';
weak_LI_filename = '/Users/slundquist/Documents/workspace/BIDS/output/BIDS_Clone_weak_LI.pvp';

filename = norm_NLI_filename;
make_roc_curves;
norm_NLI_p_set = p_set;
norm_NLI_AUC = AUC

filename = norm_LI_filename;
make_roc_curves;
norm_LI_p_set = p_set;
norm_LI_AUC = AUC

filename = weak_NLI_filename;
make_roc_curves;
weak_NLI_p_set = p_set;
weak_NLI_AUC = AUC

filename = weak_LI_filename;
make_roc_curves;
weak_LI_p_set = p_set;
weak_LI_AUC = AUC

filename = strong_NLI_filename;
make_roc_curves;
strong_NLI_p_set = p_set;
strong_NLI_AUC = AUC

filename = strong_LI_filename;
make_roc_curves;
strong_LI_p_set = p_set;
strong_LI_AUC = AUC


figure
hold on
plot([0,1],[0,1],'k')

plot(weak_NLI_p_set(1,:),weak_NLI_p_set(2,:),'LineStyle', ':', 'Color', [1 .7 .7])
plot(weak_LI_p_set(1,:),weak_LI_p_set(2,:),'Color',[.5 0 0])

plot(norm_NLI_p_set(1,:),norm_NLI_p_set(2,:),'LineStyle', ':', 'Color', [.7 1 .7])
plot(norm_LI_p_set(1,:),norm_LI_p_set(2,:),'Color',[0 .5 0])

plot(strong_NLI_p_set(1,:),strong_NLI_p_set(2,:),'LineStyle', ':', 'Color', [.7 .7 1])
plot(strong_LI_p_set(1,:),strong_LI_p_set(2,:),'Color',[0 0 .5])

%text(.05, .95, ['Area under Roc Curve: ', num2str(AUC)], 'Color', 'k');
hold off
xlim([0 1])
ylim([0 1])

legend('Chance',...
   ['Weak NLI (', num2str(weak_NLI_AUC), ')'], ['Weak LI (', num2str(weak_LI_AUC), ')'],...
   ['Norm NLI (', num2str(norm_NLI_AUC), ')'], ['Norm LI (', num2str(norm_LI_AUC), ')'],...
   ['Strong NLI (', num2str(strong_NLI_AUC), ')'] , ['Strong LI (', num2str(strong_LI_AUC), ')'],...
   'Location','SouthEast')

ylabel('Probability of Detection')
xlabel('Probability of False Alarm')
title('ROC Plot for BIDS nodes')

print([output_path, 'ROC.png'])

x = [.6 .8 .9];
y_NLI = [weak_NLI_AUC norm_NLI_AUC strong_NLI_AUC];
y_LI = [weak_LI_AUC norm_LI_AUC strong_LI_AUC];

figure
hold on
plot(x, y_NLI, 'r');
plot(x, y_LI, 'b');
hold off
legend('No Lateral Interaction', 'Lateral Interaction', 'Location', 'SouthEast');
xlabel('Signal Strength');
ylabel('Area under ROC curve');
xlim([0.6 .9])
ylim([0.6 1])
title('Signal Strength vs Area under ROC');

print([output_path, 'Strength_vs_AUC.png'])
