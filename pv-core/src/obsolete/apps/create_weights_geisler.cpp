/*
 * create_weights.cpp
 *
 *  Created on: Nov 13, 2008
 *      Author: rasmussn
 */

#include "src/include/pv_common.h"
#include "src/connections/HyPerConn.hpp"
#include "src/io/io.h"

#include <assert.h>
#include <string.h>

#define LOCAL_INPUT_PATH "input/"
const char * infile = LOCAL_INPUT_PATH "geisler.bin";

static PVPatch ** createPatches(int numBundles, int nx, int ny, int nf)
{
   PVPatch ** patches = (PVPatch**) malloc(numBundles*sizeof(PVPatch*));

   for (int i = 0; i < numBundles; i++) {
      patches[i] = pvpatch_new(nx, ny, nf);
   }

   return patches;
}

static int deletePatches(int numBundles, PVPatch ** patches)
{
   for (int i = 0; i < numBundles; i++) {
      pvpatch_delete(patches[i]);
   }
   free(patches);

   return 0;
}

int main(int argc, char * argv[])
{
   FILE * fp;

   const char * inFilename = infile;
   const char * outFilename = "weights.bin";

   int i, err = 0;
   int append = 0;              // only write one time step

   if (argc > 1) {
      inFilename = argv[1];
   }

   // header information
   const int numParams = 7;
   int params[numParams];
   int nParams;
   int nxp, nyp, nfp, numNeurons;
   int minVal, maxVal;
   int numInPatches;

   fp = pv_open_binary(inFilename, &nParams, &nxp, &nyp, &nfp);
   pv_read_binary_params(fp, numParams, params);
   minVal = params[4];
   maxVal = params[5];
   numInPatches = params[6];

   //int numItems = nxp*nyp*nfp;

   PVPatch ** inPatches = PV::HyPerConn::createPatches(numInPatches, nxp, nyp, nfp);
   PVPatch ** wPatches  = createPatches(numNeurons, nxp, nyp, nfp);

   err = pv_read_patches(fp, nfp, minVal, maxVal, inPatches, numInPatches);

   for (i = 0; i < numNeurons; i += nfp) {
      for (int f = 0; f < nfp; f++) {
         wPatches[i+f]->data = inPatches[f]->data;
      }
   }
   pv_close_binary(fp);


   // output the weight patches

   err = pv_write_patches(outFilename, append,
                          nxp, nyp, nfp, minVal, maxVal,
                          numNeurons, wPatches);

   PV::HyPerConn::deletePatches(numNeurons, inPatches);
   deletePatches(numNeurons, wPatches);

   return err;
}


