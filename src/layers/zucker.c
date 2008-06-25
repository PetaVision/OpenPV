#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>

#include <math.h>
#include <stdlib.h>

static inline int update_V(int n, float phi[], float V[], float f[]);
static inline int update_f(int n, float V[], float f[]);

void update_phi(int nc, int np, float phi_c[], float xc[], float yc[],
     float thc[], float xp[], float yp[], float thp[], float fp[]);

/**
 * Constructor for a Zucker layer (PVLayer).
 * 
 * This layer just contains the firing events for an input image (currently a circle with clutter).
 * 
 */
PVLayer* pv_new_layer_zucker(PVHyperCol* hc, int index, int nx, int ny, int no)
  {
    PVLayer* l = pv_new_layer(hc, index, nx, ny, no);
    return l;
  }

int pv_layer_begin_update(PVLayer* l, int neighbor_index, int time_index)
  {
    int k;

    float* x = l->x;
    float* y = l->y;
    float* o = l->o;

    float* phi = l->phi;
    float* f = l->f;

    for (k = 0; k < N; k += CHUNK_SIZE)
      {
        update_phi(CHUNK_SIZE, N, &phi[k], &x[k], &y[k], &o[k], x, y, o, f);
        // if (DEBUG) fprintf(stderr, "  update chunk %d k=%d %f\n", k/CHUNK_SIZE, k, phi[0]);
      }

    // if (DEBUG) printf("[%d] update_partial_state: eventmask is %p, hc_id is %d\n",
    //                   s->comm_id, s->events[hc].event, hc);

    return 0;
  }

/**
 * Add feed forward contribution to partial membrane potential
 */
int pv_layer_add_feed_forward(PVLayer* l, PVLayer* llow, int neighbor_index, int time_index)
  {
    int i;
    
    // TODO - add lateral feeds (need list of connections)
    int n = l->n_neurons;
    float w = 1.0;
    
    float* phi = l->phi;
    float* fl  = llow->f;

    for (i = 0; i < n; i++)
      {
        phi[i] += w*fl[i];
      }
    
    return 0;
  }

/**
 * Finish updating a neuron layer
 */
int pv_layer_finish_update(PVLayer* l, int time_index)
  {
    update_V(N, l->phi, l->V, l->f);
    update_f(l->n_neurons, l->V, l->f);
    return 0;
  }

/**
 * update the partial sums for membrane potential
 *
 * nc is the number of neurons to process in this chunk
 * np is the total number of neurons on this processor (size of event mask fp)
 */
void update_phi(int nc, int np, float phi_c[], float xc[], float yc[],
     float thc[], float xp[], float yp[], float thp[], float fp[])
  {
    int i, j, ii, jj;

    for (i = 0; i < nc; i+=NO)
      { /* loop over incoming neuron stream */
        for (j = 0; j < np; j+=NO)
          { /* loop over all stored neurons to form pairs */

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

            for (jj = 0; jj < NO; jj++)
              {
                chi = thp[j+jj] - (atanx2 - thc[i]);
                chi = chi + 360.0f;
                chi = fmodf(chi, 180.0f);

                if (chi >= 90.0f)
                  chi = 180.0f - chi;

                gt = expf(-chi*chi/SIG_C_P_x2);

                /* cer inhibition */
                const float INHIB_FRACTION = 0.85;
                ww = COCIRC_SCALE*gd*gt*2.0*(gt - INHIB_FRACTION);
                ww = (ww < 0.0) ? ww*INHIBIT_SCALE : ww;

                w[jj] = ww;
                w[jj+NO] = ww;
              } // for jj

            /* end weight calculation */

            for (ii = 0; ii < NO; ii++)
              {

                /*** restrict d2 to band around radius of cocircle through points i,j ****/
                const float R_CIRCLE = 36 / (float) 4; // NX*s->n_cols / (float) 4; 
                const float MIN_DENOM = 1.0E-10;//1.0E-10;
                const float SIG_C_R_x2 = 1.0; // tolerance from target cocircular radius: R_CIRCLE
                float r_cocirc;
                //      r_cocirc = d2 / (2 * dy);  //from trig if thc[i] == 0 degrees
                //      r_cocirc = d2 / ( 2 * ( dy * cos(thc[i]) - dx * sin(thc[i] ) ) ); // for any thc[i]
                /*        int ii = 0;
                 ii = 0; */
                float z = ( 2 * (dy * cos(thc[ii]) - dx * sin(thc[ii]) ) );
                float sgnz = z / (MIN_DENOM + fabs(z) );
                r_cocirc = d2 * sgnz / (MIN_DENOM + fabs(z) ); // fix denominator == 0
                gr = 1.0;//exp( -pow( (r_cocirc) - R_CIRCLE, 2 ) / SIG_C_R_x2 );

                for (jj = 0; jj < NO; jj++)
                  {
                    phi_c[i+ii] = phi_c[i+ii] + w[ii+jj]*gr*fp[j+jj];
                  }
              }
          } // for j

      } // for i

  }

/**
 * update the membrane potential
 *
 * n is the number of neurons to process in this chunk
 * phi is the partial membrane potential
 * V is the membrane potential
 * f is the firing event mask
 * I is the input image
 */
static inline int update_V(int n, float phi[], float V[], float f[])
  {
    int i;
    float r = 0.0;
    // float Vth_inh= V_TH_0_INH;

    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;

    for (i = 0; i < n; i++)
      {

        /* inhibition layer, uses same zucker weights */

        // TODO - noise, image contribution?
        // if (rand()*INV_RAND_MAX < NOISE_FREQ_INH)
        //   {
        //     r = NOISE_AMP_INH * 2 * ( rand() * INV_RAND_MAX - 0.5 );
        //   }

        // H[i] += DT_d_TAU_INH*(r + phi[i] - H[i]);
        // h[i] = ((H[i] - Vth_inh) > 0.0) ? 1.0 : 0.0;
        // H[i] -= h[i]*H[i]; // reset cells that fired

        if (rand()*INV_RAND_MAX < NOISE_FREQ)
          {
            //r = NOISE_AMP * 2 * ( rand() * INV_RAND_MAX - 0.5 );
            r = 0.0;
            // TODO - if noise only from image add to feed forward contribution
            // if (I[i] > 0.0) r = I[i];
            //      printf("adding noise, r = %f, phi = %f, %d\n", r, phi[i], i);
          }

        phi[i] -= COCIRC_SCALE*f[i]; // remove self excitation

        V[i] += DT_d_TAU*(r + phi[i] - V[i] /*- INHIBIT_AMP*h[i]*/);
        phi[i] = 0.0;
      }
    
    return 0;
  }

/**
 * update firing event mask
 *
 * n is the number of neurons to process in this chunk
 * V is the membrane potential
 * f is the firing event mask
 * I is the input image
 */
static inline int update_f(int n, float V[], float f[])
  {
    int i;
    float Vth= V_TH_0;

    for (i = 0; i < n; i++)
      {
        f[i] = ((V[i] - Vth) > 0.0) ? 1.0 : 0.0;
        V[i] -= f[i]*V[i]; // reset cells that fired
      }
    
    return 0;
  }
