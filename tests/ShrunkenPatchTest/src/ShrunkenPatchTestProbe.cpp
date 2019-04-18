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
ShrunkenPatchTestProbe::ShrunkenPatchTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize_base();
   initialize(name, hc);
}

int ShrunkenPatchTestProbe::initialize_base() { return PV_SUCCESS; }

int ShrunkenPatchTestProbe::initialize(const char *name, HyPerCol *hc) {
   correctValues = NULL;
   int status    = StatsProbe::initialize(name, hc);
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
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "nxpShrunken", &nxpShrunken);
   return;
}

void ShrunkenPatchTestProbe::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "nypShrunken", &nypShrunken);
   return;
}

/**
 * @time
 * @l
 */
Response::Status ShrunkenPatchTestProbe::outputState(double timed) {
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
      correctValues = (float *)malloc((size_t)nx * sizeof(float));

      int xScaleLog2 = getTargetLayer()->getCLayer()->xScale;

      if (xScaleLog2 >= 0) {
         Fatal().printf(
               "%s: layer \"%s\" must have nxScale > 1.\n", getDescription_c(), l->getName());
      }
      int cell_size = (int)nearbyintf(powf(2.0f, -xScaleLog2));
      int kx0       = (loc->kx0) / cell_size;
      FatalIf(!(kx0 * cell_size == loc->kx0), "Test failed.\n");
      int half_cell_size = cell_size / 2;
      FatalIf(!(half_cell_size * 2 == cell_size), "Test failed.\n");
      int num_half_cells = nx / half_cell_size;
      FatalIf(!(num_half_cells * half_cell_size == nx), "Test failed.\n");
      int cells_in_patch = nxpShrunken / cell_size;
      if (nxpShrunken != cells_in_patch * cell_size) {
         Fatal().printf(
               "ShrunkenPatchTestProbe \"%s\" error: nxpShrunken must be an integer multiple of "
               "layer \"%s\" nxScale=%d.\n",
               name,
               l->getName(),
               cell_size);
      }
      int nxp_size_parity = cells_in_patch % 2;

      int idx = 0;
      for (int hc = 0; hc < num_half_cells; hc++) {
         int m               = 2 * ((hc + 1 - nxp_size_parity) / 2) + nxp_size_parity;
         float correct_value = kx0 + 0.5f * (float)m;
         for (int k = 0; k < half_cell_size; k++) {
            correctValues[idx++] = correct_value;
         }
      }
      FatalIf(!(idx == nx), "Test failed.\n");
   }
   FatalIf(!(correctValues != NULL), "Test failed.\n");

   auto status = StatsProbe::outputState(timed);
   if (!Response::completed(status)) {
      return status;
   }
   float tol = 1e-4f;

   const float *buf = getTargetLayer()->getLayerData();

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
            Fatal().printf(
                  "%s: Incorrect value %f (should be %f) in process %d, x=%d, y=%d, f=%d\n",
                  l->getDescription_c(),
                  (double)buf[kex],
                  (double)correctValues[x],
                  parent->columnId(),
                  x,
                  y,
                  f);
         }
      }
   }

   return Response::SUCCESS;
}

ShrunkenPatchTestProbe::~ShrunkenPatchTestProbe() { free(correctValues); }
}
