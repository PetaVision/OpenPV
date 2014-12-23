pv_mex(1);   % init

pv_mex(2, [0,36*36*8*4, 36, 36, 4]); % Retina
pv_mex(3, [0, 2], [0, I_MAX, 0, 0, 0, 0]);

pv_mex(2, [1,36*36*8*4,36,36,4]); % V1E
pv_mex(3, [1, 1], [
 V_TH_0, NOISE_FREQ, NOISE_AMP, COCIRC_SCALE, DT_d_TAU, MIN_V ]);

pv_mex(2, [2,36*36*8*4,36,36,4]); % V1I
pv_mex(3, [2, 1], [
V_TH_0_INH, NOISE_FREQ_INH, NOISE_AMP_INH, SCALE_GAP, DT_d_TAU_INH, MIN_H ]);

pv_mex(4, [0,1,0,0,4], [
1.0, 0, 1.0]);

pv_mex(5,5); % run for a bit