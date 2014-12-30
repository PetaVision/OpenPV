#include "pv.h"

#include <stdio.h>
#include <math.h>

/**
 * update the partial sums for membrane potential
 *
 * @nc is the number of neurons to process in this chunk
 * @np is the total number of neurons on this processor (size of event mask fp)
 * @phi_c
 * @xc
 * @yc
 * @xp
 * @xpoffset
 * @yp
 * @ypoffset
 * @fp
 */
void update_phi(int nc, int np, float phi_c[], float xc[], float yc[], float xp[],
                float xpoffset, float yp[], float ypoffset, float fp[])
{
   int i, j, ii, jj;
   int k;
   float th[NTH];

   for (ii = 0; ii < NTH; ii++)
      th[ii] = ii * DTH;

   for (i = 0; i < nc; i += NTH) { /* loop over incoming neuron stream */
      for (j = 0, k = 0; j < np; j += NTH, k++) { /* loop over all stored neurons to form pairs */

         /* Zucker weight calculation */

         float dx, dy, d2, gd, gt, ww;
         float w[2 * NTH];
         float atanx2;
         float chi;

         dx = xp[k] + xpoffset - xc[i];
         dy = yp[k] + ypoffset - yc[i];
         d2 = dx * dx + dy * dy;
         gd = exp(-d2 / SIG_C_D_x2);

         atanx2 = RAD_TO_DEG_x2 * atan2f(dy, dx);

         for (ii = 0; ii < NTH; ii++) {
            chi = th[ii] - atanx2;
            chi = chi + 360.0f;

            // there is apparently a bug in fmodf on the SPU,
            // so the below does not work:
            /*        chi = fmodf(chi,180.0f);*/
            // alternate implementation:
            // note chi is in range [0..1080]
            chi = (chi >= 720.0f ? chi - 720.0f : chi);
            chi = (chi >= 360.0f ? chi - 360.0f : chi);
            chi = (chi >= 180.0f ? chi - 180.0f : chi);

            if (chi >= 90.0f) chi = 180.0f - chi;

            gt = exp(-chi * chi / SIG_C_P_x2);

            /* cer inhibition */
            ww = COCIRC_SCALE * gd * (2.0 * gt - 1.0);
            ww = (ww < 0.0) ? ww * INHIBIT_SCALE : ww;

            w[ii] = ww;
            w[ii + NTH] = ww;
         } // for ii

         /* end weight calculation */

         for (ii = 0; ii < NTH; ii++) {
            for (jj = 0; jj < NTH; jj++) {
               phi_c[i + ii] = phi_c[i + ii] + w[ii + jj] * fp[j + jj];
            } // for jj
         } // for ii
      } // for j

   } // for i

}
