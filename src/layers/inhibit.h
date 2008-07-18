#ifndef INHIBIT_H_
#define INHIBIT_H_

int inhibit_update(PVLayer*l);
void update_phi(int nc, int np, float phi_h[], float xc[], float yc[],
		  float thc[], float xp[], float yp[], float thp[], float hp[], int boundary, float sig_d2, float sig_p2, float scale);

#endif /*INHIBIT_H_*/
