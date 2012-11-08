/*
 * OjaKernelSpikeRateProbe.cpp
 *
 *  Created on: Nov 5, 2012
 *      Author: pschultz
 */

#include "OjaKernelSpikeRateProbe.hpp"

namespace PV {

OjaKernelSpikeRateProbe::OjaKernelSpikeRateProbe(const char * probename, const char * filename, HyPerConn * conn) {
   initialize_base();
   initialize(probename, filename, conn);
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

int OjaKernelSpikeRateProbe::initialize(const char * probename, const char * filename, HyPerConn * conn) {
   HyPerCol * hc = conn->getParent();
   PVParams * params = hc->parameters();
   targetConn = conn;
   targetOjaKernelConn = dynamic_cast<OjaKernelConn *>(conn);
   if (targetOjaKernelConn == NULL) {
      if (conn->getParent()->columnId()==0) {
         fprintf(stderr, "LCATraceProbe error: connection \"%s\" must be an LCALIFLateralConn.\n", conn->getName());
      }
      abort();
   }
   xg = params->value(probename, "x");
   yg = params->value(probename, "y");
   feature = params->value(probename, "f", 0, /*warnIfAbsent*/true);
   isInputRate = params->value(probename, "isInputRate") != 0.0;
   const PVLayerLoc * loc;
   if (isInputRate) {
      arbor = params->value(probename, "arbor", 0, /*warnIfAbsent*/true);
      loc = targetOjaKernelConn->preSynapticLayer()->getLayerLoc();
   }
   else {
      loc = targetOjaKernelConn->postSynapticLayer()->getLayerLoc();
   }
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
      if( filename ) {
         const char * outputdir = hc->getOutputPath();
         if( strlen(outputdir) + strlen(filename) + 2 > PV_PATH_MAX ) {
            fprintf(stderr, "BaseConnectionProbe: output filename \"%s/%s\" too long.  Exiting.\n",outputdir,filename);
            exit(EXIT_FAILURE);
         }
         char path[PV_PATH_MAX];
         sprintf(path, "%s/%s", outputdir, filename);

         fp = fopen(path, "w");
         if( fp == NULL )  {
            fprintf(stderr, "BaseConnectionProbe error opening \"%s\" for writing: %s.\n", path, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
      else {
         fp = stdout;
      }
   }
   else {
      fp = NULL;
   }
   conn->insertProbe(this);

   return PV_SUCCESS;
}

int OjaKernelSpikeRateProbe::outputState(double timed) {
   if (fp!=NULL) {
      if (isInputRate) {
         fprintf(fp, "Connection \"%s\", t=%f: arbor %d, x=%d, y=%d, f=%d, input integrated rate=%f\n", targetOjaKernelConn->getName(), timed, arbor, xg, yg, feature, *spikeRate);
      }
      else {
         fprintf(fp, "Connection \"%s\", t=%f: x=%d, y=%d, f=%d, output integrated rate=%f\n", targetOjaKernelConn->getName(), timed, xg, yg, feature, *spikeRate);
      }
   }
   return PV_SUCCESS;
}

OjaKernelSpikeRateProbe::~OjaKernelSpikeRateProbe()
{
}

} /* namespace PV */
