#include "GateSumPoolTestBuffer.hpp"

namespace PV {

GateSumPoolTestBuffer::GateSumPoolTestBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   ANNActivityBuffer::initialize(name, params, comm);
}

void GateSumPoolTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   // Do update state of ANN Layer first
   ANNActivityBuffer::updateBufferCPU(simTime, deltaTime);
   if (simTime <= 0.0) {
      return;
   }

   // Grab layer size
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;

   bool isCorrect = true;
   // Grab the activity layer of current layer
   for (int b = 0; b < loc->nbatch; b++) {
      const float *A = mBufferData.data() + b * getBufferSize();
      // We only care about restricted space, but iY and iX are extended
      for (int iY = loc->halo.up; iY < ny + loc->halo.up; iY++) {
         for (int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++) {
            for (int iFeature = 0; iFeature < nf; iFeature++) {
               int ext_idx = kIndex(
                     iX,
                     iY,
                     iFeature,
                     nx + loc->halo.lt + loc->halo.rt,
                     ny + loc->halo.dn + loc->halo.up,
                     nf);

               float actualvalue = A[ext_idx];

               int xval = (iX + loc->kx0 - loc->halo.lt) / 2;
               int yval = (iY + loc->ky0 - loc->halo.up) / 2;
               FatalIf(xval < 0 or xval >= loc->nxGlobal, "Test failed.\n");
               FatalIf(yval < 0 or yval >= loc->nyGlobal, "Test failed.\n");

               float expectedvalue;
               expectedvalue = iFeature * 64 + yval * 16 + xval * 2 + 4.5;
               expectedvalue *= 4;

               if (fabsf(actualvalue - expectedvalue) >= 1e-4f) {
                  ErrorLog() << "Connection " << name << " Mismatch at (" << iX << "," << iY
                             << ") : actual value: " << actualvalue
                             << " Expected value: " << expectedvalue
                             << ".  Discrepancy is a whopping " << actualvalue - expectedvalue
                             << "!  Horrors!"
                             << "\n";
                  isCorrect = false;
               }
            }
         }
      }
   }
   if (!isCorrect) {
      MPI_Barrier(mCommunicator->communicator()); // If there is an error,
      // make sure that MPI doesn't kill the run before process 0 reports the error.
      exit(EXIT_FAILURE);
   }
}

} /* namespace PV */
