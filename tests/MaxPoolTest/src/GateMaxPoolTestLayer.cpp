#include "GateMaxPoolTestLayer.hpp"

namespace PV {

GateMaxPoolTestLayer::GateMaxPoolTestLayer(const char *name, HyPerCol *hc) {
   ANNLayer::initialize(name, hc);
}

Response::Status GateMaxPoolTestLayer::updateState(double timef, double dt) {

   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;

   bool isCorrect = true;
   for (int b = 0; b < loc->nbatch; b++) {
      float *GSynExt = getChannel(CHANNEL_EXC) + b * getNumNeurons(); // gated
      float *GSynInh = getChannel(CHANNEL_INH) + b * getNumNeurons(); // gt

      // Grab the activity layer of current layer
      // We only care about restricted space
      int numActive = 0;
      for (int k = 0; k < getNumNeurons(); k++) {
         if (GSynExt[k]) {
            numActive++;
            if (GSynExt[k] != GSynInh[k]) {
               ErrorLog() << "Connection " << name << " Mismatch at batch " << b << " neuron " << k
                          << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k]
                          << ".\n";
               isCorrect = false;
            }
         }
      }

      // Must be 25% active
      float percentActive = (float)numActive / getNumNeurons();
      if (percentActive != 0.25f) {
         Fatal() << "Percent active for " << name << " is " << percentActive
                 << ", where expected is .25 at timestep " << timef << " for batch " << b << "\n";
      }
      FatalIf(!(percentActive == 0.25f), "Test failed.\n");
   }

   if (!isCorrect) {
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

} /* namespace PV */
