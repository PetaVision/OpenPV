/*
 * create_weights.cpp
 *
 *  Created on: Nov 13, 2008
 *      Author: rasmussn
 */

#include "src/include/pv_common.h"
#include "src/connections/HyPerConn.hpp"
#include <math.h>
#include <stdio.h>

// TODO
// parameters
// - strength (integrated over patch)

#define LOCAL_INPUT_PATH "input/"

#define NXW 8
#define NYW 8

#define R2_MAX               12
#define GAUSS_WEIGHT_SCALE   10.0
#define ASPECT_RATIO         4.0
#define SIGMA_EDGE           2.0
#define SIGMA_THETA          DTH
#define SIGMA_DIST_COCIRC    8.0
#define SIGMA_DIST_FEEDBACK  3.0
#define THETA_OFFSET         (PI/NO/2)

const char * fileRetinaToV1exc      = LOCAL_INPUT_PATH "connRetinaToV1exc_8x8.bin";
const char * fileRetinaToV1inh      = LOCAL_INPUT_PATH "connRetinaToV1inh_8x8.bin";
const char * fileV1inhToV1excLeft   = LOCAL_INPUT_PATH "connV1inhToV1excLeft_8x8.bin";
const char * fileV1inhToV1excRight  = LOCAL_INPUT_PATH "connV1inhToV1excRight_8x8.bin";
const char * fileV1excToV1excCocirc = LOCAL_INPUT_PATH "connV1excToV1excCocirc_8x8.bin";

int main(int argc, char * argv[])
{
   const char * filename = fileRetinaToV1exc;

   int err = 0;
   size_t size, count;

   // TODO - use command line arguments

   // pre and post synaptic layers have same density for now
   int xScale = 0;
   int yScale = 0;

   const float aspect   = ASPECT_RATIO;
   const float sigmaR   = SIGMA_EDGE;
   // const float sigmaTh  = SIGMA_THETA;
   const float r2Max    = R2_MAX;
   const float strength = GAUSS_WEIGHT_SCALE;

   const size_t dim[3] = {NO, NXW, NYW};  // nf, nx, ny

   const float nf = (float) dim[0];
   const float nx = (float) dim[1];
   const float ny = (float) dim[2];

   const int numBundles = 1;

   FILE * fd = fopen(filename, "wb");
   if (fd == NULL) {
      fprintf(stderr, "%s: ERROR opening file %s\n", argv[0], filename);
      return 1;
   }

   PVPatch ** wPatches = PV::HyPerConn::createPatches(numBundles, nx, ny, nf);

   // initialize weights

   for (int i = 0; i < numBundles; i++) {
      int numFlanks = 0;
      float shift   = 0;
      int kPre = i;
      int no = 0;
      float rotate = 1.0;

      // TODO - fixme, the interface changed and correctness needs to be checked
//      PV::HyPerConn::gauss2DCalcWeights(wPatches[i], kPre, no, xScale, yScale,
//                                        numFlanks, shift, rotate,
//                                        aspect, sigmaR, r2Max, strength);

   }

   // output the weight patches

   count = numBundles;
   size  = sizeof(PVPatch) + nx*ny*nf*sizeof(float);

   if ( fwrite(dim,    sizeof(size_t), 3, fd) != 3 ) err = 2;
   if ( fwrite(&size,  sizeof(size_t), 1, fd) != 1 ) err = 2;
   if ( fwrite(&count, sizeof(size_t), 1, fd) != 1 ) err = 2;

   for (unsigned int i = 0; i < count; i++) {
      if ( fwrite(wPatches[i], size, 1, fd) != 1) {
         fprintf(stderr, "create_weights: ERROR writing patch %d\n", i);
         return 2;
      }
    }
   fclose(fd);

   PV::HyPerConn::deletePatches(numBundles, wPatches);

   return err;
}

void preDefinedWeights(int numBundles, PVPatch ** wPatches)
{
   for (int i = 0; i < numBundles; i++) {
      float * w = wPatches[i]->data;

      int nx = (int) wPatches[i]->nx;
      int ny = (int) wPatches[i]->ny;
      int nf = (int) wPatches[i]->nf;

      // WARNING - this assumes 8x8 with 1 feature

      *w++ = 0.0; *w++ = 0.0; *w++ = 0.0; *w++ = 0.1; *w++ = 0.1; *w++ = 0.0; *w++ = 0.0; *w++ = 0.0;
      *w++ = 0.0; *w++ = 0.0; *w++ = 0.1; *w++ = 0.4; *w++ = 0.4; *w++ = 0.1; *w++ = 0.0; *w++ = 0.0;
      *w++ = 0.0; *w++ = 0.1; *w++ = 0.4; *w++ = 0.8; *w++ = 0.8; *w++ = 0.4; *w++ = 0.1; *w++ = 0.0;
      *w++ = 0.1; *w++ = 0.4; *w++ = 0.8; *w++ = 1.0; *w++ = 1.0; *w++ = 0.8; *w++ = 0.4; *w++ = 0.1;
      *w++ = 0.1; *w++ = 0.4; *w++ = 0.8; *w++ = 1.0; *w++ = 1.0; *w++ = 0.8; *w++ = 0.4; *w++ = 0.1;
      *w++ = 0.0; *w++ = 0.1; *w++ = 0.4; *w++ = 0.8; *w++ = 0.8; *w++ = 0.4; *w++ = 0.1; *w++ = 0.0;
      *w++ = 0.0; *w++ = 0.0; *w++ = 0.1; *w++ = 0.4; *w++ = 0.4; *w++ = 0.1; *w++ = 0.0; *w++ = 0.0;
      *w++ = 0.0; *w++ = 0.0; *w++ = 0.0; *w++ = 0.1; *w++ = 0.1; *w++ = 0.0; *w++ = 0.0; *w++ = 0.0;

      w = wPatches[i]->data;
      float sum = 0;
      for (int ii = 0; ii < nx*ny*nf; ii++) sum += w[ii];
      for (int ii = 0; ii < nx*ny*nf; ii++) w[ii] /= sum;
   }
}
