#ifndef INHIBIT_H_
#define INHIBIT_H_

int inhibit_update(PVLayer*l);
void update_phi(int start_idx, int nc, int np, float phi_h[], 
		eventtype_t hp[], int boundary, float sig_d2, float sig_p2, float scale,
		float inhib_fraction, float inhibit_scale);

#endif /*INHIBIT_H_*/
