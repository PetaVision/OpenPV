/*
 * LCAProbe.cpp
 *
 *  Created on: Oct 2, 2012
 *      Author: pschultz
 */

#include "LCAProbe.hpp"

namespace PV {

LCAProbe::LCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
      PointProbe(filename, layer, xLoc, yLoc, fLoc, msg) {
   initLCAProbe(filename, layer, xLoc, yLoc, fLoc, msg);
}

LCAProbe::LCAProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
      PointProbe(layer, xLoc, yLoc, fLoc, msg) {
   initLCAProbe(NULL, layer, xLoc, yLoc, fLoc, msg);
}

LCAProbe::~LCAProbe() {
}

int LCAProbe::initLCAProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) {
   // PointProbe's initialization is done during call to PointProbe's constructor from LCAProbe's constructor.
   // This should probably be changed to agree with what's done in HyPerConn and HyPerLayer
   return 0;
}

int LCAProbe::writeState(double timed, HyPerLayer * l, int k, int kex)
{
   assert(fp);
   LCALayer * lca_layer = dynamic_cast<LCALayer *>(l);
   assert(lca_layer != NULL);

   const pvdata_t * V = lca_layer->getV();
   const pvdata_t * activity = lca_layer->getLayerData();
   const pvdata_t * stim = lca_layer->getStimulus();

   fprintf(fp, "%s t=%.1f k=%d kex=%d V=%6.3f A=%6.3f Gsyn=%6.3f\n",
           msg, timed, k, kex, V[k], activity[kex], stim[k]);
   fflush(fp);
   return 0;
}

} /* namespace PV */
