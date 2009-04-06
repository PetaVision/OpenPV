/*
 * gabor.c
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

// Scraps of code for the Gabor receptive fields from
// LGN to V1.

// Meant for explicit
// inclusion in other .c files for inline efficiency.
//

#include "gabor.h"
#include "../include/pv_common.h"

#include <stdlib.h>
#include <math.h>

inline int gabor_calcWeight(gabor_params *params, float* prePos, float* postPos,
      float *ww);

// Helper function to get 3-dimensional (x,y,theta) coordinates from
// a index.
static inline void getPos3(int idx, float* coords)
{
   coords[DIMY] = (idx / (NX * V1S1_NO));
   coords[DIMX] = ((idx / V1S1_NO) % NX);
   coords[DIMO] = (idx % V1S1_NO);
}

// Helper function to get 2-dimensional (x,y) coordinates from idx
static inline void getPos2(int idx, float* coords)
{
   coords[DIMY] = (idx / (1 * NX));
   coords[DIMX] = ((idx / 1) % NX);
   coords[DIMO] = (idx % 1);
}

// Calc responses to ideal stimuli to normalize the gabors in order
// to account for pixel aliasing.
int gabor_calc_normalize(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, gabor_params* params, float *weightsF, float *weightsN,
      int *numPre)
{
   int i;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float *tempPhi, *tempf;
   float usePBCsave = GABOR_PARAMS(G_usePBC);
   float GABOR_WEIGHT_SCALE_save = GABOR_PARAMS(GABOR_WEIGHT_SCALE);
   float weightsFtmp[NO], weightsNtmp[NO];
   int theta;

   weightsF[0] = 1.0; // prevent recursion
   GABOR_PARAMS(GABOR_WEIGHT_SCALE) = 1.0; //temporary for calc_norm

   // Create a tempPhi--not really necessary. TODO: use the
   // layer's, the clear it afterwards
   tempPhi = (float*) calloc(post->numNeurons, sizeof(float));
   tempf = (float*) calloc(pre->numNeurons, sizeof(float));

   GABOR_PARAMS(G_usePBC) = 1.0;

   // Calc the response to all ones
   for (i = 0; i < post->numNeurons; i++)
      numPre[i] = 0;
   for (i = 0; i < post->numNeurons; i++)
      tempPhi[i] = 0.0;
   for (i = 0; i < pre->numNeurons; i++)
      tempf[i] = 1.0;

   gabor_rcv(pre, post, tempPhi, nActivity, tempf, params);

   for (i = 0; i < NO; i++)
      // could use any X,Y here. Pick 0,0.
      weightsNtmp[i] = tempPhi[0 * NX * NO + i] / numPre[0 * NX * NO + i];

   GABOR_PARAMS(G_usePBC) = 0.0; // turn off just in case

   // Loop for each orientation and determine the ideal response
   int y0 = NY / 2;
   int x0 = NX / 2;
   postPos[DIMY] = y0;
   postPos[DIMX] = x0;

   for (theta = 0; theta < NO; theta++) {

      postPos[DIMO] = theta;

      for (i = 0; i < post->numNeurons; i++)
         numPre[i] = 0;
      for (i = 0; i < post->numNeurons; i++)
         tempPhi[i] = 0.0;
      for (i = 0; i < pre->numNeurons; i++)
         tempf[i] = 0.0;

      int preIndex, numPreIdeal = 0;
      // build the stimulus w/maximal response: ones where gabor>0
      for (preIndex = 0; preIndex < pre->numNeurons; preIndex++) {
         float weight;

         getPos2(preIndex, prePos);
         if (gabor_calcWeight(params, prePos, postPos, &weight) && fabs(weight) > exp(
               -GABOR_PARAMS(G_R2) / pow(GABOR_PARAMS(G_SIGMA), 2))) {
            numPreIdeal++;
            tempf[preIndex] = 1.0;
         }
      }

      // Feed the stimulus into gabor_rcv to find the output response
      gabor_rcv(pre, post, tempPhi, nActivity, tempf, params);
      //ASSERT( numPreIdeal == numPre(y0*NX+x0)*NO+theta);
      weightsFtmp[theta] = tempPhi[(y0 * NX + x0) * NO + theta] - numPre[(y0 * NX + x0)
            * NO + theta] * weightsNtmp[theta];
   }

   for (i = 0; i < NO; i++)
      weightsN[i] = weightsNtmp[i];
   for (i = 0; i < NO; i++)
      weightsF[i] = weightsFtmp[i];

   GABOR_PARAMS(G_usePBC) = usePBCsave;
   GABOR_PARAMS(GABOR_WEIGHT_SCALE) = GABOR_WEIGHT_SCALE_save;

   free(tempf);
   free(tempPhi);

   return 0;
}

// Calculate the "weight" between the two neurons.
inline int gabor_calcWeight(gabor_params *params, float* prePos, float* postPos,
      float *ww)
{
   // return the weight
   float sigma, gamma, lambda, phi, theta;
   int b0, b1, b2, b3, b4, b5, b6;
   float xp, yp;
   float dx, dy, d2;
   int featVec;

   // Get the Euclidean distance between the two points
   dx = prePos[DIMX] - postPos[DIMX];
   dy = prePos[DIMY] - postPos[DIMY];

   // Apply periodic boundary conditions
   if (GABOR_PARAMS(G_usePBC)) {
      dx = fabs(dx) > NX / 2 ? -(dx / fabs(dx)) * (NX - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > NY / 2 ? -(dy / fabs(dy)) * (NY - fabs(dy)) : dy;
   }

   d2 = dx * dx + dy * dy;

   if (d2 > GABOR_PARAMS(G_R2)) {
      *ww = 0;
      return 0;
   }

   featVec = (int) postPos[DIMO];

   // MIT Gabor code
   b6 = (featVec / 64) % 2;
   b5 = (featVec / 32) % 2;
   b4 = (featVec / 16) % 2;
   b3 = (featVec / 8) % 2;
   b2 = (featVec / 4) % 2;
   b1 = (featVec / 2) % 2;
   b0 = (featVec / 1) % 2;

   sigma = GABOR_PARAMS(G_SIGMA);
   gamma = GABOR_PARAMS(GAMMA_BASE) + GABOR_PARAMS(GAMMA_MULT) * b6;
   lambda = GABOR_PARAMS(LAMBDA_BASE) + GABOR_PARAMS(LAMBDA_MULT) * b5;
   phi = 0.0 + (PI / 2.0) * (2 * b4 + b3);
   theta = (PI / 8.0) * (4 * b2 + 2 * b1 + b0);

   // Right now we are not using most of the bits, just the 8 orientations.
   phi = 0;

   // Swap x and y so that a horizonal edge is detected at 0 degs.
   xp = dx * cos(theta) + dy * sin(theta);
   yp = -1.0 * dx * sin(theta) + dy * cos(theta);

   // Calculate the weight based on a Gabor using the given parameters, for this orientation
   *ww = exp(GABOR_PARAMS(GABOR_WEIGHT_SCALE_EXP) * -1.0 * (xp * xp + gamma * gamma * yp * yp) / (sigma
         * sigma)) * cos(phi + 2.0 * PI * yp / lambda);

   if (fabs(*ww) > exp(-GABOR_PARAMS(G_R2) / pow(GABOR_PARAMS(G_SIGMA), 2))) return 1;
   else return 0;
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
int gabor_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
      void *params)
{
   int i, j;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float weight;
   static float normF[8] = { -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
   static float normN[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
   static int numPre[NX * NY * NO];

   if (normF[0] == -1.0) gabor_calc_normalize(pre, post, phi, nActivity, fActivity,
         (gabor_params*) params, normF, normN, numPre);

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++) {
      if (fActivity[j] == 0.0) continue; // optimization: skip 0 inputs (spiking or continuous)

      // Determine presynaptic neuron's features
      // TODO: Need to translate the pre vs. post column
      // TODO: Temporarily skip over the extra orientation dimension in LGN
      getPos2(j, prePos);

      // For each neuron in the postsynaptic patch
      for (i = 0; i < post->numNeurons; i++) {
         // Determine postsynaptic feature vector
         getPos3(i, postPos);

         // Call the weight calculation handler:
         // Normalize by the output for the ideal stimulus
         // response to uniform input = 0
         // normF[abs(4-postPos[DIMO]) % 4] // alternative
         if (gabor_calcWeight((gabor_params*) params, prePos, postPos, &weight)) {
            numPre[i]++;
            phi[i] += GABOR_PARAMS(GABOR_WEIGHT_SCALE) * (weight - normN[(int) postPos[DIMO]]) / fabs(
                  normF[(int) postPos[DIMO]]);
         }
      } // for each input neuron
   }
   return 0;
}

