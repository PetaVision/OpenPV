#include "ComparisonLayer.hpp"

namespace PV {

ComparisonLayer::ComparisonLayer(const char *name, PVParams *params, Communicator *comm) {
   ANNLayer::initialize(name, params, comm);
}

Response::Status ComparisonLayer::checkUpdateState(double timef, double dt) {

   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;

   float const *GSynExt = mLayerInput->getChannelData(CHANNEL_EXC); // gated
   float const *GSynInh = mLayerInput->getChannelData(CHANNEL_INH); // gt

   bool isCorrect = true;
   // Grab the activity layer of current layer
   // We only care about restricted space
   for (int k = 0; k < getNumNeurons(); k++) {
      if (GSynExt[k] != GSynInh[k]) {
         ErrorLog() << "Connection " << name << " Mismatch at " << k
                    << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k]
                    << ".\n";
         isCorrect = false;
      }
   }

   if (!isCorrect) {
      exit(-1);
   }
   return Response::SUCCESS;
}

} /* namespace PV */
