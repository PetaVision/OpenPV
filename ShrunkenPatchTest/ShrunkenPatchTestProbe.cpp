/*
 * StatsProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "ShrunkenPatchTestProbe.hpp"
#include "../PetaVision/src/include/pv_arch.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include "../PetaVision/src/io/PVParams.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
ShrunkenPatchTestProbe::ShrunkenPatchTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initShrunkenPatchTestProbe(probename, filename, layer, msg);
}

/**
 * @type
 * @msg
 */
ShrunkenPatchTestProbe::ShrunkenPatchTestProbe(const char * probename, HyPerLayer * layer, const char * msg)
: StatsProbe()
{
   initShrunkenPatchTestProbe(probename, NULL, layer, msg);
}

int ShrunkenPatchTestProbe::initShrunkenPatchTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg) {
   correctValues = NULL;
   int status = StatsProbe::initStatsProbe(filename, layer, BufActivity, msg);
   if (probename==NULL) {
      fprintf(stderr, "ShrunkenPatchTestProbe error in rank %d process: probename must be set.\n", layer->getParent()->columnId());
      exit(EXIT_SUCCESS);
   }
   probeName = strdup(probename);
   if (probeName==NULL) {
      fprintf(stderr, "ShrunkenPatchTestProbe \"%s\" error in rank %d process: unable to copy probename.\n", probename, layer->getParent()->columnId());
      exit(EXIT_SUCCESS);
   }
   PVParams * params = getTargetLayer()->getParent()->parameters();
   readNxpShrunken(params);
   readNypShrunken(params);

   return status;
}

void ShrunkenPatchTestProbe::readNxpShrunken(PVParams * params) {
   nxpShrunken=params->value(probeName, "nxpShrunken");
   return;
}

void ShrunkenPatchTestProbe::readNypShrunken(PVParams * params) {
   nypShrunken=params->value(probeName, "nypShrunken");
   return;
}


/**
 * @time
 * @l
 */
int ShrunkenPatchTestProbe::outputState(double timed) {
   HyPerLayer * l = getTargetLayer();
   const PVLayerLoc * loc = l->getLayerLoc();
   int num_neurons = l->getNumNeurons();

   // nxpShrunken must be an integer multiple of the layer's nxScale, and nxScale must be a positive integral power of 2.
   // The correct values of the layer activity is a function of its column index, that depends on nxpShrunken.
   // If nxpShrunken is an odd multiple of nxScale, the patch is not really shrunken, and then the correct values
   // of the layer activity are [<0.5> <0.5> <1.5> <1.5> <2.5> <2.5> ...], where angle brackets indicate that the given
   // value is repeated nxScale/2 times.
   //
   // If nxpShrunken is an even multiple, the correct values of the layer activity are
   // [ <0.0> <1.0> <1.0> <2.0> <2.0> ...]
   //
   // This assumes the connection with l as the post-synaptic layer has a pre-synaptic layer with nxScale=1.
   // There isn't a convenient way for a ShrunkenPatchTestProbe object to ensure that that's the case.

   if (correctValues==NULL) {
      int nx=loc->nx;
      correctValues = (pvdata_t *) malloc((size_t) nx*sizeof(pvdata_t));

      int xScaleLog2 = getTargetLayer()->getCLayer()->xScale;

      if (xScaleLog2>=0) {
         fprintf(stderr, "ShrunkenPatchTestProbe \"%s\" error: layer \"%s\" must have nxScale > 1.\n", probeName, l->getName());
         abort();
      }
      int cell_size = (int) nearbyintf(powf(2.0f, -xScaleLog2));
      int kx0 = (loc->kx0)/cell_size;
      assert(kx0*cell_size == loc->kx0);
      int half_cell_size = cell_size/2;
      assert(half_cell_size*2==cell_size);
      int num_half_cells = nx/half_cell_size;
      assert(num_half_cells*half_cell_size==nx);
      int cells_in_patch = nxpShrunken/cell_size;
      if (nxpShrunken != cells_in_patch*cell_size) {
         fprintf(stderr, "ShrunkenPatchTestProbe \"%s\" error: nxpShrunken must be an integer multiple of layer \"%s\" nxScale=%d.\n", probeName, l->getName(), cell_size);
         abort();
      }
      int nxp_size_parity = cells_in_patch % 2;

      int idx=0;
      for (int hc=0; hc<num_half_cells; hc++) {
         int m = 2 * ((hc+1-nxp_size_parity)/2) + nxp_size_parity;
         pvdata_t correct_value = kx0 + 0.5f * (pvdata_t) m;
         for (int k=0; k<half_cell_size; k++) {
            correctValues[idx++] = correct_value;
         }
      }
      assert(idx==nx);
   }
   assert(correctValues!=NULL);

   int status = StatsProbe::outputState(timed);
   double tol = 1e-4f;

   const pvdata_t * buf = getTargetLayer()->getLayerData();

   if (timed>=3.0f) {
      for (int k=0; k<num_neurons; k++) {
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         int x = kxPos(k,loc->nx, loc->ny, loc->nf);
         if (fabs(buf[kex]-correctValues[x])>tol) {
            int y = kyPos(k,loc->nx, loc->ny, loc->nf);
            int f = featureIndex(k,loc->nx,loc->ny,loc->nf);
            fprintf(stderr, "Layer \"%s\": Incorrect value %f (should be %f) in process %d, x=%d, y=%d, f=%d\n",
                  l->getName(), buf[kex], correctValues[x], getTargetLayer()->getParent()->columnId(), x, y, f);
            abort();
         }
      }
   }

   return status;
}

ShrunkenPatchTestProbe::~ShrunkenPatchTestProbe() {
   free(probeName);
   free(correctValues);
}

}
