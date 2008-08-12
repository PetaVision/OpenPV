#ifndef ZUCKER_H_
#define ZUCKER_H_

PVLayer* pv_new_layer_zucker(PVHyperCol* hc, int index, int nx, int ny, int no, int nk);
inline int update_V(int n, float phi[],float phi_h[], float V[], float f[], float dt_d_tau, float v_min, float noise_freq, float noise_amp, float scale);
inline int update_f(int n, float V[], float f[], float Vth);
void debug_filer (char* path,eventtype_t *h, float *phi1, float *phi2, int time, int n);

#endif /*ZUCKER_H_*/
