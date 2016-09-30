/*
 * StatsProbe.cpp
 *
 *  Created on:
 *      Author: garkenyon
 */

#include "ShrunkenPatchTestProbe.hpp"
#include <include/pv_arch.h>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
ShrunkenPatchTestProbe::ShrunkenPatchTestProbe(const char *probename, HyPerCol *hc) : StatsProbe() {
   initShrunkenPatchTestProbe_base();
   initShrunkenPatchTestProbe(probename, hc);
}

int ShrunkenPatchTestProbe::initShrunkenPatchTestProbe_base() { return PV_SUCCESS; }

int ShrunkenPatchTestProbe::initShrunkenPatchTestProbe(const char *probename, HyPerCol *hc) {
   correctValues = NULL;
   int status    = StatsProbe::initStatsProbe(probename, hc);
   return status;
}

int ShrunkenPatchTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_nxpShrunken(ioFlag);
   ioParam_nypShrunken(ioFlag);
   return status;
}

void ShrunkenPatchTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

void ShrunkenPatchTestProbe::ioParam_nxpShrunken(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValueRequired(ioFlag, getName(), "nxpShrunken", &nxpShrunken);
   return;
}

void ShrunkenPatchTestProbe::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   getParent()->parameters()->ioParamValueRequired(ioFlag, getName(), "nypShrunken", &nypShrunken);
   return;
}

/**
 * @time
 * @l
 */
int ShrunkenPatchTestProbe::outputState(double timed) {
   HyPerLayer *l         = getTargetLayer();
   const PVLayerLoc *loc = l->getLayerLoc();
   int num_neurons       = l->getNumNeurons();

   // nxpShrunken must be an integer multiple of the layer's nxScale, and nxScale must be a positive
   // integral power of 2.
   // The correct values of the layer activity is a function of its column index, that depends on
   // nxpShrunken.
   // If nxpShrunken is an odd multiple of nxScale, the patch is not really shrunken, and then the
   // correct values
   // of the layer activity are [<0.5> <0.5> <1.5> <1.5> <2.5> <2.5> ...], where angle brackets
   // indicate that the given
   // value is repeated nxScale/2 times.
   //
   // If nxpShrunken is an even multiple, the correct values of the layer activity are
   // [ <0.0> <1.0> <1.0> <2.0> <2.0> ...]
   //
   // This assumes the connection with l as the post-synaptic layer has a pre-synaptic layer with
   // nxScale=1.
   // There isn't a convenient way for a ShrunkenPatchTestProbe object to ensure that that's the
   // case.

   if (correctValues == NULL) {
      int nx        = loc->nx;
      correctValues = (pvdata_t *)malloc((size_t)nx * sizeof(pvdata_t));

      int xScaleLog2 = getTargetLayer()->getCLayer()->xScale;

      if (xScaleLog2 >= 0) {
         pvError().printf(
               "%s: layer \"%s\" must have nxScale > 1.\n", getDescription_c(), l->getName());
      }
      int cell_size = (int)nearbyintf(powf(2.0f, -xScaleLog2));
      int kx0       = (loc->kx0) / cell_size;
      pvErrorIf(!(kx0 * cell_size == loc->kx0), "Test failed.\n");
      int half_cell_size = cell_size / 2;
      pvErrorIf(!(half_cell_size * 2 == cell_size), "Test failed.\n");
      int num_half_cells = nx / half_cell_size;
      pvErrorIf(!(num_half_cells * half_cell_size == nx), "Test failed.\n");
      int cells_in_patch = nxpShrunken / cell_size;
      if (nxpShrunken != cells_in_patch * cell_size) {
         pvError().printf(
               "ShrunkenPatchTestProbe \"%s\" error: nxpShrunken must be an integer multiple of "
               "layer \"%s\" nxScale=%d.\n",
               probeName,
               l->getName(),
               cell_size);
      }
      int nxp_size_parity = cells_in_patch % 2;

      int idx = 0;
      for (int hc = 0; hc < num_half_cells; hc++) {
         int m                  = 2 * ((hc + 1 - nxp_size_parity) / 2) + nxp_size_parity;
         pvdata_t correct_value = kx0 + 0.5f * (pvdata_t)m;
         for (int k = 0; k < half_cell_size; k++) {
            correctValues[idx++] = correct_value;
         }
      }
      pvErrorIf(!(idx == nx), "Test failed.\n");
   }
   pvErrorIf(!(correctValues != NULL), "Test failed.\n");

   int status = StatsProbe::outputState(timed);
   float tol  = 1e-4f;

   const pvdata_t *buf = getTargetLayer()->getLayerData();

   if (timed >= 3.0) {
      for (int k = 0; k < num_neurons; k++) {
         int kex = kIndexExtended(
               k,
               loc->nx,
               loc->ny,
               loc->nf,
               loc->halo.lt,
               loc->halo.rt,
               loc->halo.dn,
               loc->halo.up);
         int x = kxPos(k, loc->nx, loc->ny, loc->nf);
         if (fabsf(buf[kex] - correctValues[x]) > tol) {
            int y = kyPos(k, loc->nx, loc->ny, loc->nf);
            int f = featureIndex(k, loc->nx, loc->ny, loc->nf);
            pvError().printf(
                  "%s: Incorrect value %f (should be %f) in process %d, x=%d, y=%d, f=%d\n",
                  l->getDescription_c(),
                  (double)buf[kex],
                  (double)correctValues[x],
                  getTargetLayer()->getParent()->columnId(),
                  x,
                  y,
                  f);
         }
      }
   }

   return status;
}

ShrunkenPatchTestProbe::~ShrunkenPatchTestProbe() {
   // free(probeName);
   free(correctValues);
}
}
