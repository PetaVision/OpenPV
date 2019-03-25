#include "GateMaxPoolTestBuffer.hpp"

namespace PV {

GateMaxPoolTestBuffer::GateMaxPoolTestBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void GateMaxPoolTestBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   GSynAccumulator::initialize(name, params, comm);
}

void GateMaxPoolTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   if (simTime <= 0.0) {
      return;
   }

   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;

   bool isCorrect = true;
   for (int b = 0; b < loc->nbatch; b++) {
      float const *GSynExt = mLayerInput->getBufferData(b, CHANNEL_EXC); // gated
      float const *GSynInh = mLayerInput->getBufferData(b, CHANNEL_INH); // gt

      // Compare the excitatory and inhibitory inputs; they should be equal.
      int numActive = 0;
      for (int k = 0; k < getBufferSize(); k++) {
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

      // 25% of the neurons should be active
      float percentActive = (float)numActive / getBufferSize();
      if (percentActive != 0.25f) {
         Fatal() << "Percent active for " << name << " is " << percentActive
                 << ", where expected is .25 at timestep " << simTime << " for batch " << b << "\n";
      }
      FatalIf(!(percentActive == 0.25f), "Test failed.\n");
   }

   if (!isCorrect) {
      exit(EXIT_FAILURE);
   }
}

} /* namespace PV */
