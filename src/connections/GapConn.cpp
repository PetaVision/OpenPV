/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"
#include "../layers/LIFGap.hpp"
#include "../normalizers/NormalizeGap.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

GapConn::GapConn() { initialize_base(); }

GapConn::GapConn(const char *name, HyPerCol *hc) {
   initialize_base();
   GapConn::initialize(name, hc);
}

GapConn::~GapConn() {}

int GapConn::initialize_base() {
   initNormalizeFlag = false;
   return PV_SUCCESS;
}

int GapConn::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

void GapConn::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      channel = CHANNEL_GAP;
      parent->parameters()->handleUnnecessaryParameter(name, "channelCode", (int)CHANNEL_GAP);
   }
}

void GapConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // Default of true for sharedWeights for GapConns was deprecated Aug 11, 2014.
   // This default was chosen for backwards compatibility because GapConn used to require
   // sharedWeights be true.
   // Now GapConn can be used with or without shared weights, so eventually the default will false
   // as it is for other HyPerConns.
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &sharedWeights, true /*default*/, true /*warn if absent*/);
   if (ioFlag == PARAMS_IO_READ && !parent->parameters()->present(name, "sharedWeights")) {
      sharedWeights = true;
      if (parent->columnId() == 0) {
         WarnLog().printf(
               "%s: sharedWeights defaults to true for GapConns, but the default may be changed to "
               "false in a future release, to be consistent with other HyPerConns.\n",
               getDescription_c());
      }
      return;
   }
   HyPerConn::ioParam_sharedWeights(ioFlag);
}

void GapConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   // Default of normalizeSum for normalizeMethod for GapConns was deprecated Aug 11, 2014.
   // This default was chosen for backwards compatibility because GapConn used to require
   // normalizeMethod be normalizeSum.
   // Now GapConn can be normalized using any method, so eventually the default will be removed and
   // the parameter required as is for other HyPerConns.
   if (ioFlag == PARAMS_IO_READ && !parent->parameters()->stringPresent(name, "normalizeMethod")) {
      normalizeMethod = strdup("normalizeSum");
      GapConn *conn   = this;
      normalizer      = new NormalizeGap(name, parent);
      if (parent->columnId() == 0) {
         WarnLog().printf(
               "%s: normalizeMethod defaults to normalizeSum for GapConns, but this parameter may "
               "be required in a future release, to be consistent with other HyPerConns.\n",
               getDescription_c());
      }
      return;
   }
   HyPerConn::ioParam_normalizeMethod(ioFlag);
}

int GapConn::allocateDataStructures() {
   HyPerLayer *postHyPerLayer = this->postSynapticLayer();
   LIFGap *postLIFGap         = dynamic_cast<LIFGap *>(postHyPerLayer);
   if (postLIFGap == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: postsynaptic layer must be a LIFGap or LIFGap-derived layer.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   int status = HyPerConn::allocateDataStructures();

   return status;
}

int GapConn::deliver() {
   int status = PV_SUCCESS;

   // Check if updating from post perspective
   HyPerLayer *pre = preSynapticLayer();
   int numArbors   = numberOfAxonalArborLists();

   for (int arbor = 0; arbor < numArbors; arbor++) {
      int delay        = getDelay(arbor);
      PVLayerCube cube = pre->getPublisher()->createCube(delay);
      cube.numItems /= cube.loc.nbatch;
      // hack; should make sure deliver*Perspective* methods expect numItems to include batching.
      if (!getUpdateGSynFromPostPerspective()) {
#ifdef PV_USE_CUDA
         if (getReceiveGpu()) {
            status = deliverPresynapticPerspectiveGPU(&cube, arbor);
            // No need to update GSyn since it's already living on gpu
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPresynapticPerspective(&cube, arbor);
#ifdef PV_USE_CUDA
            // CPU updated gsyn, need to update gsyn
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      else {
#ifdef PV_USE_CUDA
         if (getReceiveGpu()) {
            status = deliverPostsynapticPerspectiveGPU(&cube, arbor);
            // GSyn already living on GPU
            post->setUpdatedDeviceGSynFlag(false);
         }
         else
#endif
         {
            status = deliverPostsynapticPerspective(&cube, arbor);
#ifdef PV_USE_CUDA
            // CPU updated gsyn, need to update on GPU
            post->setUpdatedDeviceGSynFlag(true);
#endif
         }
      }
      pvAssert(status == PV_SUCCESS || status == PV_BREAK);
      if (status == PV_BREAK) {
         break; // Breaks out of arbor loop
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
