/*
 * LCAProbe.cpp
 *
 *  Created on: Oct 2, 2012
 *      Author: pschultz
 */

#include "LCAProbe.hpp"

namespace PV {

LCAProbe::LCAProbe() {
   initLCAProbe_base();
   // Derived classes should call initLCAProbe from their initializer.
}

LCAProbe::LCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
      PointProbe(filename, layer, xLoc, yLoc, fLoc, msg) {
   initLCAProbe_base();
   initLCAProbe(filename, layer, xLoc, yLoc, fLoc, msg);
}

LCAProbe::LCAProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
      PointProbe(layer, xLoc, yLoc, fLoc, msg) {
   initLCAProbe_base();
   initLCAProbe(NULL, layer, xLoc, yLoc, fLoc, msg);
}

LCAProbe::~LCAProbe() {
}

int LCAProbe::initLCAProbe_base() {
   return PV_SUCCESS;
}

int LCAProbe::initLCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) {

   return PointProbe::initPointProbe(filename, layer, xLoc, yLoc, fLoc, msg);
}

int LCAProbe::writeState(double timed, HyPerLayer * l, int k, int kex)
{
   assert(outputstream && outputstream->fp);
   LCALayer * lca_layer = dynamic_cast<LCALayer *>(l);
   assert(lca_layer != NULL);

   const pvdata_t * V = lca_layer->getV();
   const pvdata_t * activity = lca_layer->getLayerData();
   const pvdata_t * stim = lca_layer->getStimulus();

   fprintf(outputstream->fp, "%s t=%.1f k=%d kex=%d V=%6.3f A=%6.3f Gsyn=%6.3f\n",
           msg, timed, k, kex, V[k], activity[kex], stim[k]);
   fflush(outputstream->fp);
   return 0;
}

} /* namespace PV */
