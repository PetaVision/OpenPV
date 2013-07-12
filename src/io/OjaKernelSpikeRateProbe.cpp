/*
 * OjaKernelSpikeRateProbe.cpp
 *
 *  Created on: Nov 5, 2012
 *      Author: pschultz
 */

#include "OjaKernelSpikeRateProbe.hpp"

namespace PV {

OjaKernelSpikeRateProbe::OjaKernelSpikeRateProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   initialize(probename, hc);
}

OjaKernelSpikeRateProbe::OjaKernelSpikeRateProbe()
{
   initialize_base();
}

int OjaKernelSpikeRateProbe::initialize_base() {
   targetOjaKernelConn = NULL;
   spikeRate = NULL;
   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::initialize(const char * probename, HyPerCol * hc) {
   BaseConnectionProbe::initialize(probename, hc);
   PVParams * params = hc->parameters();
   xg = params->value(probename, "x");
   yg = params->value(probename, "y");
   feature = params->value(probename, "f", 0, /*warnIfAbsent*/true);
   isInputRate = params->value(probename, "isInputRate") != 0.0;
   if (isInputRate) {
      arbor = params->value(probename, "arbor", 0, /*warnIfAbsent*/true);
   }
   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::allocateProbe() {
   targetOjaKernelConn = dynamic_cast<OjaKernelConn *>(getTargetConn());
   if (targetOjaKernelConn == NULL) {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "LCATraceProbe error: connection \"%s\" must be an LCALIFLateralConn.\n", getTargetConn()->getName());
      }
      abort();
   }
   HyPerLayer * targetLayer = NULL;
   if (isInputRate) {
      targetLayer = targetOjaKernelConn->preSynapticLayer();
   }
   else {
      targetLayer = targetOjaKernelConn->postSynapticLayer();
   }
   const PVLayerLoc * loc = targetLayer->getLayerLoc();
   int x_local = xg - loc->kx0;
   int y_local = yg - loc->ky0;
   bool inBounds = (x_local >= 0 && x_local < loc->nx && y_local >= 0 && y_local < loc->ny);
   if(inBounds ) { // if inBounds
      int krestricted = kIndex(x_local, y_local, feature, loc->nx, loc->ny, loc->nf);
      if (isInputRate) {
         int kextended = kIndexExtended(krestricted, loc->nx, loc->ny, loc->nf, loc->nb);
         spikeRate = &targetOjaKernelConn->getInputFiringRate(arbor)[kextended];
      }
      else {
         spikeRate = &targetOjaKernelConn->getOutputFiringRate()[krestricted];
      }
   }
   else {
      stream = NULL;
   }
   getTargetConn()->insertProbe(this);

   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::outputState(double timed) {
   if (stream!=NULL) {
      if (isInputRate) {
         fprintf(stream->fp, "Connection \"%s\", t=%f: arbor %d, x=%d, y=%d, f=%d, input integrated rate=%f\n", targetOjaKernelConn->getName(), timed, arbor, xg, yg, feature, *spikeRate);
      }
      else {
         fprintf(stream->fp, "Connection \"%s\", t=%f: x=%d, y=%d, f=%d, output integrated rate=%f\n", targetOjaKernelConn->getName(), timed, xg, yg, feature, *spikeRate);
      }
   }
   return PV_SUCCESS;
}

OjaKernelSpikeRateProbe::~OjaKernelSpikeRateProbe()
{
}

} /* namespace PV */
