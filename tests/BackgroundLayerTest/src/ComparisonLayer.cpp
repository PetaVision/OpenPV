#include "ComparisonLayer.hpp"

namespace PV {

ComparisonLayer::ComparisonLayer(const char *name, PVParams *params, Communicator const *comm) {
   ANNLayer::initialize(name, params, comm);
}

Response::Status ComparisonLayer::checkUpdateState(double timef, double dt) {
   float const *GSynExt = mLayerInput->getChannelData(CHANNEL_EXC); // gated
   float const *GSynInh = mLayerInput->getChannelData(CHANNEL_INH); // gt

   bool isCorrect = true;
   // Grab the activity layer of current layer
   // We only care about restricted space
   for (int k = 0; k < getNumNeurons(); k++) {
      if (GSynExt[k] != GSynInh[k]) {
         ErrorLog() << "Connection " << getName() << " Mismatch at " << k
                    << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k]
                    << ".\n";
         isCorrect = false;
      }
   }

   if (!isCorrect) {
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

} /* namespace PV */
