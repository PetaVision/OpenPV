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
   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseConnectionProbe::ioParamsFillGroup(ioFlag);
   ioParam_x(ioFlag);
   ioParam_y(ioFlag);
   ioParam_f(ioFlag);
   ioParam_isInputRate(ioFlag);
   ioParam_arbor(ioFlag);
   return status;
}

void OjaKernelSpikeRateProbe::ioParam_x(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "x", &xg);
}

void OjaKernelSpikeRateProbe::ioParam_y(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "y", &yg);
}

void OjaKernelSpikeRateProbe::ioParam_f(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "f", &feature);
}

void OjaKernelSpikeRateProbe::ioParam_isInputRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "isInputRate", &isInputRate, false/*default value*/);
}

void OjaKernelSpikeRateProbe::ioParam_arbor(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "isInputRate"));
   if (isInputRate) {
      parent->ioParamValue(ioFlag, name, "arbor", &arbor, 0);
   }
}

int OjaKernelSpikeRateProbe::allocateDataStructures() {
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
         int kextended = kIndexExtended(krestricted, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         spikeRate = &targetOjaKernelConn->getInputFiringRate(arbor)[kextended];
      }
      else {
         spikeRate = &targetOjaKernelConn->getOutputFiringRate()[krestricted];
      }
   }
   else {
      outputstream = NULL;
   }
   //This is now being done in BaseConnectionProbe
   //getTargetConn()->insertProbe(this);

   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::outputState(double timed) {
   if (outputstream!=NULL) {
      if (isInputRate) {
         fprintf(outputstream->fp, "Connection \"%s\", t=%f: arbor %d, x=%d, y=%d, f=%d, input integrated rate=%f\n", targetOjaKernelConn->getName(), timed, arbor, xg, yg, feature, *spikeRate);
      }
      else {
         fprintf(outputstream->fp, "Connection \"%s\", t=%f: x=%d, y=%d, f=%d, output integrated rate=%f\n", targetOjaKernelConn->getName(), timed, xg, yg, feature, *spikeRate);
      }
   }
   return PV_SUCCESS;
}

OjaKernelSpikeRateProbe::~OjaKernelSpikeRateProbe()
{
}

} /* namespace PV */
