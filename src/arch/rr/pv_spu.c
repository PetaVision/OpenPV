#include "pv.h"

#include <stdio.h>
#include <math.h>

void start_clock();
float stop_clock();

/**
 * update the partial sums for membrane potential
 *
 * @nc is the number of neurons to process in this chunk
 * @np is the total number of neurons on this processor (size of event mask fp)
 * @phi_c
 * @xc
 * @yc
 * @thc
 * @xp
 * @yp
 * @thp
 * @fp
 */
void update_phi(int nc, int np, float phi_c[], float xc[], float yc[], float thc[],
		float xp[], float yp[], float thp[], float fp[])
{
  int i, j, ii, jj;

  for (i = 0; i < nc; i+=NO) {	/* loop over incoming neuron stream */
    for (j = 0; j < np; j+=NO) {	/* loop over all stored neurons to form pairs */

      /* Zucker weight calculation */

      float dx, dy, d2, gd, gt, gr, ww;
      float w[2*NO];
      float atanx2;
      float chi;

      dx = xp[j] - xc[i];
      dy = yp[j] - yc[i];
      d2 = dx*dx + dy*dy;
      gd = expf(-d2/SIG_C_D_x2);

      atanx2 = RAD_TO_DEG_x2*atan2f(dy,dx);

      for (jj = 0; jj < NO; jj++) {
        chi = thp[j+jj] - (atanx2 - thc[i]);
        chi = chi + 360.0f;
        chi = fmodf(chi,180.0f);

        if (chi >= 90.0f) chi = 180.0f - chi;

        gt = expf(-chi*chi/SIG_C_P_x2);

	/* cer inhibition */
		const float INHIB_FRACTION = 0.85;
		ww = COCIRC_SCALE*gd*gt*2.0*(gt - INHIB_FRACTION);
        ww = (ww < 0.0) ? ww*INHIBIT_SCALE : ww;

        w[jj] = ww;
        w[jj+NO] = ww;
      } // for jj

        /* end weight calculation */

      for (ii = 0; ii < NO; ii++) {

      /*** restrict d2 to band around radius of cocircle through points i,j ****/
      const float R_CIRCLE = 36 / (float) 4;   // NX*s->n_cols / (float) 4;
      const float MIN_DENOM = 1.0E-10;//1.0E-10;
      const float SIG_C_R_x2 = 1.0;  // tolerance from target cocircular radius: R_CIRCLE
      float r_cocirc;
      //      r_cocirc = d2 / (2 * dy);  //from trig if thc[i] == 0 degrees
      //      r_cocirc = d2 / ( 2 * ( dy * cos(thc[i]) - dx * sin(thc[i] ) ) ); // for any thc[i]
/* 	  int ii = 0;
	  ii = 0; */
	  float z = ( 2 * ( dy * cos(thc[ii]) - dx * sin(thc[ii] ) ) );
	  float sgnz = z / ( MIN_DENOM + fabs(z) );
      r_cocirc = d2 * sgnz / ( MIN_DENOM + fabs(z) ); // fix denominator == 0
	  gr = 1.0;//exp( -pow( (r_cocirc) - R_CIRCLE, 2 ) / SIG_C_R_x2 );

        for (jj = 0; jj < NO; jj++) {
          phi_c[i+ii] = phi_c[i+ii] + w[ii+jj]*gr*fp[j+jj];
        }
      }
    } // for j

  } // for i
}
