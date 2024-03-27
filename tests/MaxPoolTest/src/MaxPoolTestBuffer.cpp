#include "MaxPoolTestBuffer.hpp"

namespace PV {

MaxPoolTestBuffer::MaxPoolTestBuffer(const char *name, PVParams *params, Communicator const *comm) {
   ANNActivityBuffer::initialize(name, params, comm);
}

void MaxPoolTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   // Do update state of ANNActivityBuffer first
   ANNActivityBuffer::updateBufferCPU(simTime, deltaTime);
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
   FatalIf(nf != 3, "Test requires %s have nf=3 (instead it has %d).\n", getDescription_c(), nf);

   bool isCorrect = true;
   for (int b = 0; b < loc->nbatch; b++) {
      // Grab the activity layer of current layer
      const float *A = mBufferData.data() + b * getBufferSize();
      // We only care about restricted space
      for (int iY = loc->halo.up; iY < ny + loc->halo.up; iY++) {
         for (int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++) {
            for (int iFeature = 0; iFeature < nf; iFeature++) {
               int idx = kIndex(
                     iX,
                     iY,
                     iFeature,
                     nx + loc->halo.lt + loc->halo.rt,
                     ny + loc->halo.dn + loc->halo.up,
                     nf);
               // Input image is set up to have max values in 3rd feature dimension
               // 3rd dimension, top left is 128, bottom right is 191
               // Y axis spins fastest
               float actualvalue = A[idx];

               int xval = iX + kx0 - loc->halo.lt;
               int yval = iY + ky0 - loc->halo.up;
               // Patches on edges have same answer as previous neuron
               if (xval == 7) {
                  xval -= 1;
               }
               if (yval == 7) {
                  yval -= 1;
               }

               // modified GTK: 1/10/15, modified to test spatial max pooling over a feature plane
               // float expectedvalue = 8*xval+yval+137;
               float expectedvalue = (yval + 1) + 8 * (xval + 1) + 64 * iFeature;
               if (actualvalue != expectedvalue) {
                  ErrorLog() << "Connection " << getName() << " Mismatch at (" << iX << "," << iY
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
      exit(EXIT_FAILURE);
   }
}

} /* namespace PV */
