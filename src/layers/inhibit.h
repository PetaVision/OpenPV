#ifndef INHIBIT_H_
#define INHIBIT_H_

int inhibit_update(PVLayer*l, int time_index);

void update_phi(int nc, int np, int noc, int nop, int nkc,int nkp, float phi_h[], float xc[], float yc[],
		float thc[],float kappap[], float xp[], float yp[], float thp[], float hp[], int boundary, float sig_d2, float sig_p2, float scale, 
		float inhib_fraction, float inhibit_scale, int curve, float sig_k2, int self);
#endif /*INHIBIT_H_*/
