// ---------------------------------------------------
// Co-circularity connection routines
// ---------------------------------------------------

#include "cocirc_k.h"
#include "../include/pv_common.h"
#include "../include/neural_tuning.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define PARAMS(p) (((cocircK_params*)params)->p)
static inline int cocircK_calcWeight(cocircK_params* params, float *prePos,
                                     float* postPos, float* weight, int memoize);
int cocircK_rcv_opt(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
                    float *fActivity, void *params);

// Helper function to get 4-dimensional (x,y,theta,k) coordinates from
// a index. nk might be 1 (for inhibitory layers)
static inline void getPos(int idx, float* coords, int ny, int nx, int no, int nk)
{
   coords[DIMY] = (idx / (nx * no * nk));
   coords[DIMX] = ((idx / (no * nk)) % nx);
   coords[DIMO] = (idx / nk) % no;
   coords[DIMK] = (idx % nk);
}

static inline void getPosAdjusted(int idx, PVLayer *l, float* coords, int ny, int nx,
                                  int no, int nk)
{
   getPos(idx, coords, ny, nx, no, nk);

   // Adjust for the HyPerColumn
   coords[DIMY] = coords[DIMY] * l->loc.dy + l->yOrigin;
   coords[DIMX] = coords[DIMX] * l->loc.dx + l->xOrigin;
}

#if 0 // will try to use 4-dim version, but pass in "0" as NK
// Helper function to get 2-dimensional (x,y) coordinates from idx
static inline void getPos3(int idx, float* coords)
{
   coords[DIMY] = (idx / (NX * NO));
   coords[DIMX] = ((idx / (NO)) % NX);
   coords[DIMO] = (idx) % NO;
}

static inline void getPos3a(int idx, PVLayer *l, float* coords)
{
   getPos3(idx, coords);

   // Adjust for the HyPerColumn
   coords[DIMY] = coords[DIMY] * l->dy + l->yOrigin;
   coords[DIMX] = coords[DIMX] * l->dx + l->xOrigin;
}
#endif //0
// New, modular versions are not ready.
// Almost straight copy of the legacy versions are at the end of the file.
#if 0 //  new modular versions not ready
static inline int cocircK_calcWeight(cocircK_params* params, float *prePos,
      float* postPos, float* weight, int memoize)
{
   static float dx, dy, d2, gd;
   static float gr = 1.0;
   static float atanx2;
   static float k_cocirc, selfcorrect;

   if (memoize) {
      // Optimization: set memoize to 1 if it's known that the current
      // X,Y,O value are are the same as the previous.

      // Calculate the distance between the two neurons' x and y locations
      dx = postPos[DIMX] - prePos[DIMX];
      dy = postPos[DIMY] - prePos[DIMY];

      // If desired, apply periodic boundary conditions
      if (PARAMS(CCK_usePBC)) {
         dx = fabs(dx)> PARAMS(CCK_NX) / 2 ? -(dx / fabs(dx)) * (PARAMS(CCK_NX) - fabs(dx)) : dx; // PBCs
         dy = fabs(dy)> PARAMS(CCK_NY) / 2 ? -(dy / fabs(dy)) * (PARAMS(CCK_NY) - fabs(dy)) : dy;
      }

      d2 = dx * dx + dy * dy;

      if (d2> PARAMS(CCK_R2)) {
         *weight = PARAMS(CCK_DISTANT_VAL);
         return 0;
      }

      /*** restrict d2 to band around radius of cocircle through points i,j ***/
      float radp = (DEG_TO_RAD) * prePos[DIMO] * PARAMS(CCK_DTH);
      float dxP = (dx * cos(radp) + dy * sin(radp));
      float dyP = (dy * cos(radp) - dx * sin(radp));
      float z = (2 * dyP);

      if (d2 == 0) {   // dx==0 && dy==0
         atanx2 = prePos[DIMO] * PARAMS(CCK_DTH);
         selfcorrect = PARAMS(CCK_SELF_CORRECT) == 0 ? 1 : 0;
         k_cocirc = 0.0; // actually 1/0  -- TODO not sure about this
      }
      else {
         atanx2 = prePos[DIMO] * PARAMS(CCK_DTH) + RAD_TO_DEG_x2 * atan2f(dyP, dxP);
         selfcorrect = PARAMS(CCK_SELF_CORRECT);
         k_cocirc = fabs(z) / d2;
      }

      gr = exp(-PARAMS(CCK_USE_CURVATURE) * (k_cocirc - (prePos[DIMK] * DK))
            * (k_cocirc - (prePos[DIMK] * DK)) / PARAMS(CCK_SIG_C_CURV_x2));
      gd = expf(-d2 / PARAMS(CCK_SIG_C_D_x2));

   } // end memoized code--following code executes every time:

   float chi, gt, ww;

   chi = atanx2 - postPos[DIMO] * PARAMS(CCK_DTH); // Calc int. angle of this orienation
   chi = chi + 360.0f;
   chi = fmodf(chi, 180.0f);
   if (chi >= 90.0f)
   chi = 180.0f - chi;

   // Apply Gaussians
   gt = expf(-chi * chi / PARAMS(CCK_SIG_C_P_x2));

   // Calculate and apply connection efficacy/weight
   ww = fabs(gd * gr) * (fabs(gt) - PARAMS(CCK_INHIB_FRACTION));
   ww = (ww < 0.0) ? ww * PARAMS(CCK_INHIBIT_SCALE) : ww;

   ww = PARAMS(CCK_COCIRC_SCALE) * ww * selfcorrect;

   // TODO: Make a macro so that in release builds we don't even check the print param: just noop
   if (PARAMS(CCK_DEBUG))
   printf(
         "yxo=%3.3f %3.3f %3.3f yxo=%3.3f %3.3f %3.3f "
         "th=%f, theta'=%f, radius=%f k_cocirc=%f chi=%f yp=%f xp=%f, gr=%f self_correct=%f\n",
         prePos[DIMY], prePos[DIMX], prePos[DIMO], postPos[DIMY],
         postPos[DIMX], postPos[DIMO], 0.0, 0.0, 0.0, 0.0, chi, 0.0,
         0.0, gr, selfcorrect);

   *weight = ww;
   return 1;
}

// Pre and post have curv
int cocirc1_rcv(PVLayer* pre, PVLayer* post, int nActivity, float *activity,
      void *params)
{
   int i, j;
   float* prePos, *postPos;
   float weight;
   // TODO - take into account extended border
   float* phi = post->phi;

   prePos = (float*) malloc(sizeof(float) * pre->numDims);
   postPos = (float*) malloc(sizeof(float) * post->numDims);

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++)
   {
      if (activity[j])
      continue; // optimization: skip 0 inputs (spiking or continuous)

      // Determine presynaptic neuron's features
      // TODO: Need to translate the pre vs. post column
      getPos4(j, prePos);

      // For each neuron in the postsynaptic patch
      // Optimization: stride by NK, since we don't care about it. Later
      // copy the weights to all the curved neurons for this position.
      for (i = 0; i < post->numNeurons; i += NK)
      {
         // Determine postsynaptic feature vector
         getPos4(i, postPos);

         // Call the weight calculation handler:
         cocircK_calcWeight((cocircK_params*) params, prePos, postPos,
               &weight, i % (NK * NO) == 0);

         // Copy the weights to all the curv dimensions
         int ki;
         for (ki = 0; ki < NK; ki++) {
            // TODO - take into account extended border
            phi[i + ki] += weight;
         }
      }
   }

   free(prePos);
   free(postPos);

   return 0;
}

// Pre has curv, post does not:
int cocirc2_rcv(PVLayer* pre, PVLayer* post, int nActivity, float *activity,
      void *params)
{
   int i, j;
   float* prePos, *postPos;
   float weight;
   // TODO - take into account extended border
   float* phi = post->phi;

   prePos = (float*) malloc(sizeof(float) * pre->numDims);
   postPos = (float*) malloc(sizeof(float) * post->numDims);

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++)
   {
      if (activity[j] == 0.0)
      continue; // optimization: skip 0 inputs (spiking or continuous)

      // Determine presynaptic neuron's features
      // TODO: Need to translate the pre vs. post column
      getPos4(j, prePos);

      // For each neuron in the postsynaptic patch
      for (i = 0; i < post->numNeurons; i++)
      {
         // Determine postsynaptic feature vector
         getPos3(i, postPos);

         // Call the weight calculation handler:
         cocircK_calcWeight((cocircK_params*) params, prePos, postPos,
               &weight, i % NO == 0);

         // Just sum all connections coming into a given postsynaptic neuron:
         // TODO - take into account extended border
         phi[i] += weight;
      } // for each input neuron
   }

   free(prePos);
   free(postPos);
}

// Post has curv, pre does not:
int cocirc3_rcv(PVLayer* pre, PVLayer* post, int nActivity, float *activity,
      void *params)
{
   int i, j;
   float* prePos, *postPos;
   float weight;
   // TODO - take into account extended border
   float* phi = post->phi;

   prePos = (float*) malloc(sizeof(float) * pre->numDims);
   postPos = (float*) malloc(sizeof(float) * post->numDims);

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++)
   {
      if (activity[j] == 0.0)
      continue; // optimization: skip 0 inputs (spiking or continuous)

      // Determine presynaptic neuron's features
      // TODO: Need to translate the pre vs. post column
      getPos3(j, prePos);

      // For each neuron in the postsynaptic patch
      // Optimization: stride by NK, since we don't care about it. Later
      // copy the weights to all the curved neurons for this position.
      for (i = 0; i < post->numNeurons; i += NK)
      {
         // Determine postsynaptic feature vector
         getPos4(i, postPos);

         // Call the weight calculation handler:
         cocircK_calcWeight((cocircK_params*) params, prePos, postPos,
               &weight, i % (NO * NK) == 0);

         // Copy the weights to all the curv dimensions
         int ki;
         for (ki = 0; ki < NK; ki++) }
            // TODO - take into account extended border
            phi[i + ki] += weight;
         }

      } // for each input neuron
   }

   free(prePos);
   free(postPos);
}
#endif //0
// Calc responses to ideal stimuli to normalize the gabors in order
// to account for pixel aliasing.
int cocircK2_calc_normalize(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, cocircK_params* params, float *weightsF, int *numPre)
{
   int i;
   float *tempPhi, *tempf;
   float usePBCsave = PARAMS(CCK_usePBC);
   float CCK_SCALE_save = PARAMS(CCK_SCALE);
   float postNsave = PARAMS(CCK_POST_N);

   const int postNeuronsForCalibration = NO * NK * 2;

   // init, and prevent recursion
   for (i = 0; i < NO * NK; i++) {
      weightsF[i] = 1.0;
   }

   // TODO - take into account extended border
   tempPhi = (float*) calloc(post->numNeurons, sizeof(float));
   tempf = (float*) calloc(pre->numNeurons, sizeof(float));

   PARAMS(CCK_SCALE) = 1.0; //temporary for calc_norm
   PARAMS(CCK_usePBC) = 1.0;
   PARAMS(CCK_POST_N) = postNeuronsForCalibration; // just look at the response of 64 neurons

   // Calc the response to all ones
   for (i = 0; i < postNeuronsForCalibration; i++)
      numPre[i] = 0;
   for (i = 0; i < post->numNeurons; i++) {
      // TODO - take into account extended border
      tempPhi[i] = 0.0;
   }
   for (i = 0; i < pre->numNeurons; i++)
      tempf[i] = 1.0;

   //clock_t ticks = clock();
   cocircK2_rcv(pre, post, tempPhi, nActivity, tempf, params);
   //ticks = clock() - ticks;
   //printf("ticks: %ll\n", ticks);
   for (i = 0; i < NO * NK; i++) {
      // could use any X,Y here. Pick 0,0.
      weightsF[i] = tempPhi[0 * NX * NO * NK + i] / numPre[0 * NK * NO * NK + i];
   }

   PARAMS(CCK_usePBC) = 0.0; // turn off just in case

   PARAMS(CCK_usePBC) = usePBCsave;
   PARAMS(CCK_SCALE) = CCK_SCALE_save;
   PARAMS(CCK_POST_N) = postNsave;

   free(tempf);
   free(tempPhi);
   return 0;
}

// Calc responses to ideal stimuli to normalize the gabors in order
// to account for pixel aliasing.
int cocircK_calc_normalize(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, cocircK_params* params, float *weightsF, int *numPre)
{
   int i;
   float *tempPhi, *tempf;
   float usePBCsave = PARAMS(CCK_usePBC);
   float CCK_SCALE_save = PARAMS(CCK_SCALE);
   float postNsave = PARAMS(CCK_POST_N);

   const int postNeuronsForCalibration = NO * NK * 2;

   // init, and prevent recursion
   for (i = 0; i < NO * NK; i++) {
      weightsF[i] = 1.0;
   }

   // TODO - take into account extended border
   tempPhi = (float*) calloc(post->numNeurons, sizeof(float));
   tempf = (float*) calloc(pre->numNeurons, sizeof(float));

   PARAMS(CCK_SCALE) = 1.0; //temporary for calc_norm
   PARAMS(CCK_usePBC) = 1.0;
   PARAMS(CCK_POST_N) = postNeuronsForCalibration; // just look at the response of 64 neurons

   // Calc the response to all ones
   for (i = 0; i < postNeuronsForCalibration; i++)
      numPre[i] = 0;
   for (i = 0; i < post->numNeurons; i++) {
      // TODO - take into account extended border
      tempPhi[i] = 0.0;
   }
   for (i = 0; i < pre->numNeurons; i++)
      tempf[i] = 1.0;

   cocircK_rcv(pre, post, tempPhi, nActivity, tempf, params);
   for (i = 0; i < NO * NK; i++) {
      // could use any X,Y here. Pick 0,0.
      //weightsNtmp[i] = tempPhi[0*NX*NO*NK+i] / numPre[0*NX*NO+i];
      weightsF[i] = tempPhi[0 * NX * NO * NK + i] / numPre[0 * NK * NO * NK + i];
   }

   PARAMS(CCK_POST_N) = postNsave;
   PARAMS(CCK_usePBC) = usePBCsave;
   PARAMS(CCK_SCALE) = CCK_SCALE_save;

   free(tempf);
   free(tempPhi);
   return 0;
}

static inline int cocircK_calcWeight(cocircK_params* params, float *prePos,
      float* postPos, float* weight, int memoize)
{
   float dx, dy, d2;
   float gd, gt, gr, ww;
   float selfcorrect;
   int self2 = (PARAMS(CCK_SELF) == 0) ? 1 : 0;

   // Calc euclidean distance between neurons.
   dx = postPos[DIMX] - prePos[DIMX];
   dy = postPos[DIMY] - prePos[DIMY];

   // If desired, apply periodic boundary conditions
   if (PARAMS(CCK_usePBC)) {
      dx = fabs(dx) > PARAMS(CCK_NX) / 2 ? -(dx / fabs(dx)) * (PARAMS(CCK_NX) - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > PARAMS(CCK_NY) / 2 ? -(dy / fabs(dy)) * (PARAMS(CCK_NY) - fabs(dy)) : dy;
   }

   d2 = dx * dx + dy * dy; // euclidean distance

   // Hard limit the distance between the neurons
   if (d2 > PARAMS(CCK_BOUNDARY)) {
      *weight = 0;
      return 0;
   }

   float atanx2;
   float chi;

   /*** restrict d2 to band around radius of cocircle through points i,j ***/
   // TODO: precompute sin/cos
   float k_cocirc;
   float radp = (DEG_TO_RAD) * prePos[DIMO] * PARAMS(CCK_DTH);
   float dxP = (dx * cos(radp) + dy * sin(radp));
   float dyP = (dy * cos(radp) - dx * sin(radp));
   float z = (2 * dyP);
   k_cocirc = fabs(z) / (MIN_DENOM + d2); // fix denominator == 0

   // TODO: optimize: for CCK_CURVE (0 or 1) better to skip Gaussian?
   gr = fabs(exp(-PARAMS(CCK_CURVE) * pow((k_cocirc - fabs(prePos[DIMK] * DK)), 2) / PARAMS(CCK_SIG_K2)));

   if (dx == 0 && dy == 0) atanx2 = prePos[DIMO] * PARAMS(CCK_DTH);
   else atanx2 = prePos[DIMO] * PARAMS(CCK_DTH) + RAD_TO_DEG_x2 * atan2f(dyP, dxP);

   gd = expf(-d2 / PARAMS(CCK_SIG_D2));

   chi = atanx2 - postPos[DIMO] * PARAMS(CCK_DTH);
   chi = chi + 360.0f;
   chi = fmodf(chi, 180.0f);
   if (chi >= 90.0f) chi = 180.0f - chi;

   gt = expf(-chi * chi / PARAMS(CCK_SIG_P2));

   ww = fabs(gd * gr) * (fabs(gt) - PARAMS(CCK_INHIB_FRACTION));
   ww = (ww < 0.0) ? ww * PARAMS(CCK_INHIBIT_SCALE) : ww;
   selfcorrect = (PARAMS(CCK_SELF) * d2 / (d2 + MIN_DENOM) + self2);

   *weight = ww * selfcorrect;
   return 1;
}

// This is the modular version we'd like to someday use, but it's too slow right now...
int cocircK_rcv_modular(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, void *params)
{
   int i, j;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float weight;

   // normalization stuff:
   static float normF[8] = { -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
   static float normN[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
   static int numPre[2 * NO * NK]; // our call to uniform field only uses this much

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++) {
      if (fActivity[j] == 0.0) continue; // optimization: skip 0 inputs (spiking or continuous)

      getPos(j, prePos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_PRE_NO), PARAMS(CCK_PRE_NK));

      // For each neuron in the postsynaptic patch
      for (i = 0; i < post->numNeurons; i++) {
         getPos(i, postPos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_POST_NO), PARAMS(CCK_POST_NK));

         // Call the weight calculation handler:
         cocircK_calcWeight((cocircK_params*) params, prePos, postPos, &weight, 0);

         // Just sum all connections coming into a given postsynaptic neuron:
         numPre[i]++;
         // TODO - take into account extended border
         phi[i] += PARAMS(CCK_SCALE) * (weight - normN[(int) (postPos[DIMO] * NK + postPos[DIMK])])
               / fabs(normF[(int) (postPos[DIMO] * NK + postPos[DIMK])]);
         phi[i] += weight;
      } // for each input neuron
   }
   return 0;
}

// --------------------------------------------------------------------------
// Default rcvSynapticInput() implementation:
//
// Input: non-sparse activity input.
// Output: For each postsynaptic, neuron, sum of weights based on input activity.
//
// Algorithm: Finds each active presynaptic neurons and calculates weight for
// each post-synaptic neuron, summing weights to get a single value for each
// post-synaptic neuron.
// --------------------------------------------------------------------------

// This is a direct port of "update_phi()" from Petavision trunk.
int cocircK_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
      void *params)
{
   int i, j, ii, jj, jjj, iii;
   //int curve2 = (PARAMS(CCK_CURVE) == 0) ? 1 : 0;
   int self2 = (PARAMS(CCK_SELF) == 0) ? 1 : 0;
   float prePos[4], postPos[4]; // TODO: max dims or something?
#ifdef COK_STATS
   float phit;
   float phiAve = 0.0, phiMax = FLT_MIN, phiMin = FLT_MAX;
   float VAve = 0.0, VMax = FLT_MIN, VMin = FLT_MAX;
#endif //COK_STATS
   // normalization stuff:
   static float normF[NO * NK] = { -1.0 }; // more inits in calc_normalize!
   static float normN[NO * NK] = { 0.0 }; //more inits in calc_normalize!
   static int numPre[NX * NY * NO * NK]; // our call to uniform field only uses this much

   if (normF[0] == -1.0) cocircK_calc_normalize(pre, post, phi, nActivity, fActivity,
         (cocircK_params*) params, normF, numPre);

   for (j = 0; j < PARAMS(CCK_PRE_N); j += (PARAMS(CCK_PRE_NO) * PARAMS(CCK_PRE_NK))) { // loop over all x,y locations
      for (jj = 0; jj < (PARAMS(CCK_PRE_NO) * PARAMS(CCK_PRE_NK)); jj += PARAMS(CCK_PRE_NK)) { // loop over all orientations
         for (jjj = 0; jjj < PARAMS(CCK_PRE_NK); jjj++) {
            if (fActivity[j + jj + jjj] == 0.0) // If this neuron didn't fire, skip it.
            continue;

            getPosAdjusted(j + jj + jjj, pre, prePos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_PRE_NO), PARAMS(CCK_PRE_NK));

            for (i = 0; i < PARAMS(CCK_POST_N); i += (PARAMS(CCK_POST_NO) * PARAMS(CCK_POST_NK))) { // loop over other neurons, first by x,y

               float dx, dy, d2, gd, gt, ww;
               float selfcorrect;

               getPos(i, postPos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_POST_NO), PARAMS(CCK_POST_NK));

               // Calc euclidean distance between neurons.
               dx = postPos[DIMX] - prePos[DIMX];
               dy = postPos[DIMY] - prePos[DIMY];

               // If desired, apply periodic boundary conditions
               if (PARAMS(CCK_usePBC)) {
                  dx = fabs(dx) > PARAMS(CCK_NX) / 2 ? -(dx / fabs(dx)) * (PARAMS(CCK_NX) - fabs(dx)) : dx; // PBCs
                  dy = fabs(dy) > PARAMS(CCK_NY) / 2 ? -(dy / fabs(dy)) * (PARAMS(CCK_NY) - fabs(dy)) : dy;
               }

               d2 = dx * dx + dy * dy; // euclidean distance

               // Hard limit the distance between the neurons
               if (d2 > PARAMS(CCK_BOUNDARY)) continue;

               float gr;
               float atanx2;
               float chi;

               /*** restrict d2 to band around radius of cocircle through points i,j ***/
               // TODO: precompute sin/cos
               float k_cocirc;
               float radp = (DEG_TO_RAD) * prePos[DIMO] * PARAMS(CCK_DTH);
               float dxP = (dx * cos(radp) + dy * sin(radp));
               float dyP = (dy * cos(radp) - dx * sin(radp));
               float z = (2 * dyP);
               k_cocirc = fabs(z) / (MIN_DENOM + d2); // fix denominator == 0
               // TODO: optimize: for CCK_CURVE (0 or 1) better to skip Gaussian?
               gr = fabs(exp(-PARAMS(CCK_CURVE) * pow((k_cocirc - fabs(prePos[DIMK] * DK)), 2) / PARAMS(CCK_SIG_K2)));

               if (dx == 0 && dy == 0) atanx2 = prePos[DIMO] * PARAMS(CCK_DTH);
               else atanx2 = prePos[DIMO] * PARAMS(CCK_DTH) + RAD_TO_DEG_x2 * atan2f(dyP, dxP);

               gd = expf(-d2 / PARAMS(CCK_SIG_D2));

               for (ii = 0; ii < (PARAMS(CCK_POST_NO) * PARAMS(CCK_POST_NK)); ii += PARAMS(CCK_POST_NK)) { // now loop over each orienation

                  // TODO fixme
                  postPos[DIMO] = ii / PARAMS(CCK_POST_NK);

                  chi = atanx2 - postPos[DIMO] * PARAMS(CCK_DTH);
                  chi = chi + 360.0f;
                  chi = fmodf(chi, 180.0f);
                  if (chi >= 90.0f) chi = 180.0f - chi;

                  gt = expf(-chi * chi / PARAMS(CCK_SIG_P2));

                  ww = fabs(gd * gr) * (fabs(gt) - PARAMS(CCK_INHIB_FRACTION));
                  ww = (ww < 0.0) ? ww * PARAMS(CCK_INHIBIT_SCALE) : ww;
                  selfcorrect = (PARAMS(CCK_SELF) * d2 / (d2 + MIN_DENOM) + self2);

                  // Copy the same weight to all (or 1) postsynaptic curvatures
                  for (iii = 0; iii < PARAMS(CCK_POST_NK); iii++) {
                     float weight = ww * selfcorrect;
                     if (fabs(weight) > .001) { // parameterize this?
                        weight = PARAMS(CCK_SCALE) * (weight - normN[(int) (postPos[DIMO] * NK
                              + postPos[DIMK])]) / fabs(normF[(int) (postPos[DIMO] * NK
                              + postPos[DIMK])]);
                        // TODO - take into account extended border
                        phi[i + ii + iii] += weight;
                        numPre[i + ii + iii]++;
                     }
                     // Gather some statistics
#ifdef COK_STATS
                     phit = PARAMS(CCK_SCALE) * ww; //phi[i + ii + iii];
                     phiAve += phit;
                     if (phit < phiMin)
                     phiMin = phit;
                     if (phit> phiMax)
                     phiMax = phit;
#endif
                  }//iii
               } // ii
            } // i

         }// for jjj
      } // for jj
   } // for j

#ifdef COK_STATS
   char msg[128];
   sprintf(msg, "%d: phi: Max: %1.4f, Avg=%1.4f Min=%1.4f\n", j + jj + jjj,
         phiMax, phiAve, phiMin);
   printf(msg);
#endif //COK_STATS
   return 0;
}

// This is a direct port of "update_phi()" from Petavision trunk.
// With symmetric curvature preference connections, obviously
// only relevant with E->E.
int cocircK2_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, void *params)
{
   int i, j, ii, jj, jjj, iii;
   //int curve2 = (PARAMS(CCK_CURVE) == 0) ? 1 : 0;
   int self2 = (PARAMS(CCK_SELF) == 0) ? 1 : 0;
   float prePos[MAX_DIMS], postPos[MAX_DIMS]; // TODO: max dims or something?
#ifdef COK_STATS
   float phit;
   float phiAve = 0.0, phiMax = FLT_MIN, phiMin = FLT_MAX;
   float VAve = 0.0, VMax = FLT_MIN, VMin = FLT_MAX;
#endif //COK_STATS
   // normalization stuff:
   static float normF[NO * NK] = { -1.0 }; // more inits in calc_normalize!
   static int numPre[NX * NY * NO * NK]; // need all since this routine is generic

   if (normF[0] == -1.0) cocircK2_calc_normalize(pre, post, phi, nActivity, fActivity,
         (cocircK_params*) params, normF, numPre);

   for (j = 0; j < PARAMS(CCK_PRE_N); j += (PARAMS(CCK_PRE_NO) * PARAMS(CCK_PRE_NK))) { // loop over all x,y locations
      for (jj = 0; jj < (PARAMS(CCK_PRE_NO) * PARAMS(CCK_PRE_NK)); jj += PARAMS(CCK_PRE_NK)) { // loop over all orientations
         for (jjj = 0; jjj < PARAMS(CCK_PRE_NK); jjj++) {
            if (fActivity[j + jj + jjj] == 0.0) // If this neuron didn't fire, skip it.
            continue;

            getPosAdjusted(j + jj + jjj, pre, prePos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_PRE_NO), PARAMS(CCK_PRE_NK));

            for (i = 0; i < PARAMS(CCK_POST_N); i += (PARAMS(CCK_POST_NO) * PARAMS(CCK_POST_NK))) { // loop over other neurons, first by x,y

               float dx, dy, d2, gd, gt, ww;
               float selfcorrect;

               getPos(i, postPos, PARAMS(CCK_NY), PARAMS(CCK_NX), PARAMS(CCK_POST_NO), PARAMS(CCK_POST_NK));

               // Calc euclidean distance between neurons.
               dx = postPos[DIMX] - prePos[DIMX];
               dy = postPos[DIMY] - prePos[DIMY];

               // If desired, apply periodic boundary conditions
               if (PARAMS(CCK_usePBC)) {
                  dx = fabs(dx) > PARAMS(CCK_NX) / 2 ? -(dx / fabs(dx)) * (PARAMS(CCK_NX) - fabs(dx)) : dx; // PBCs
                  dy = fabs(dy) > PARAMS(CCK_NY) / 2 ? -(dy / fabs(dy)) * (PARAMS(CCK_NY) - fabs(dy)) : dy;
               }

               d2 = dx * dx + dy * dy; // euclidean distance

               // Hard limit the distance between the neurons
               if (d2 > PARAMS(CCK_BOUNDARY)) continue;

               float gr;
               float atanx2;
               float chi;

               /*** restrict d2 to band around radius of cocircle through points i,j ***/
               // TODO: precompute sin/cos
               float k_cocirc;
               float radp = (DEG_TO_RAD) * prePos[DIMO] * PARAMS(CCK_DTH);
               float dxP = (dx * cos(radp) + dy * sin(radp));
               float dyP = (dy * cos(radp) - dx * sin(radp));
               float z = (2 * dyP);
               k_cocirc = fabs(z) / (MIN_DENOM + d2); // fix denominator == 0
               // TODO: optimize: for CCK_CURVE (0 or 1) better to skip Gaussian?
               gr = fabs(exp(-PARAMS(CCK_CURVE) * pow((k_cocirc - fabs(prePos[DIMK] * DK)), 2) / PARAMS(CCK_SIG_K2)));

               if (dx == 0 && dy == 0) atanx2 = prePos[DIMO] * PARAMS(CCK_DTH);
               else atanx2 = prePos[DIMO] * PARAMS(CCK_DTH) + RAD_TO_DEG_x2 * atan2f(dyP, dxP);

               gd = expf(-d2 / PARAMS(CCK_SIG_D2));

               selfcorrect = (PARAMS(CCK_SELF) * d2 / (d2 + MIN_DENOM) + self2);

               for (ii = 0; ii < (PARAMS(CCK_POST_NO) * PARAMS(CCK_POST_NK)); ii += PARAMS(CCK_POST_NK)) { // now loop over each orienation

                  // TODO fixme
                  postPos[DIMO] = ii / PARAMS(CCK_POST_NK);

                  chi = atanx2 - postPos[DIMO] * PARAMS(CCK_DTH);
                  chi = chi + 360.0f;
                  chi = fmodf(chi, 180.0f);
                  if (chi >= 90.0f) chi = 180.0f - chi;

                  gt = expf(-chi * chi / PARAMS(CCK_SIG_P2));

                  // Copy the same weight to all (or 1) postsynaptic curvatures
                  for (iii = 0; iii < PARAMS(CCK_POST_NK); iii++) {
                     // TODO fixme
                     // I think this matches legacy behavior:
                     // if CCK_POST_NK is 1 (post is an inhibitory layer), then
                     // postPos[DIMK] will only be 0.0, like kappai[]. BUT, I
                     // think what we really want is for grpost to be 1 if post
                     // doesn't have curvature, which would require different logic.
                     postPos[DIMK] = iii / PARAMS(CCK_POST_NK);

                     float grpost = (exp(-PARAMS(CCK_CURVE) * ((k_cocirc - postPos[DIMK] * DK) * (k_cocirc
                           - postPos[DIMK] * DK)) / PARAMS(CCK_SIG_K2)));
                     ww = (gd * gr * grpost) * (gt - PARAMS(CCK_INHIB_FRACTION));
                     ww = (ww < 0.0) ? ww * PARAMS(CCK_INHIBIT_SCALE) : ww;

                     float weight = PARAMS(CCK_SELF) * ww * selfcorrect;

                     if (fabs(weight) > .001) { // TODO:  parameterize
                        weight = PARAMS(CCK_SCALE) * weight / fabs(normF[(int) (postPos[DIMO] * NK
                              + postPos[DIMK])]);
                        // TODO - take into account extended border
                        phi[i + ii + iii] += weight;
                        numPre[i + ii + iii]++;
                     }
                     else {
                        // Below our cutoff
                        weight = 0.0;
                     }
#ifdef COK_STATS
                     // Gather some statistics
                     phit = weight; //phi[i + ii + iii];
                     phiAve += phit;
                     if (phit < phiMin)
                     phiMin = phit;
                     if (phit> phiMax)
                     phiMax = phit;
#endif //COK_STATS
                  }//iii
               } // ii
            } // i

         }// for jjj
      } // for jj
   } // for j

#ifdef COK_STATS
   char msg[128];
   sprintf(msg, "%d: phi: Max: %1.4f, Avg=%1.4f Min=%1.4f\n", j + jj + jjj,
         phiMax, phiAve, phiMin);
   printf(msg);
#endif //COK_STATS
   return 0;
}
