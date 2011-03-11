function plotAmoeba2D3(amoeba_image, amoeba_struct, trial_ndx)

plotAmoebaPTB2(amoeba_image, amoeba_struct, trial_ndx, true);
plotAmoebaPTB2(amoeba_image, amoeba_struct, trial_ndx, false);
plotAmoebaMask(amoeba_image, amoeba_struct, trial_ndx);
plotAmoebaPTB2(amoeba_image, amoeba_struct, trial_ndx, true, true);


