#include "GateSumPoolTestLayer.hpp"

namespace PV {

GateSumPoolTestLayer::GateSumPoolTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int GateSumPoolTestLayer::updateState(double timef, double dt){
   //Do update state of ANN Layer first
   ANNLayer::updateState(timef, dt);

   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;

   bool isCorrect = true;
   //Grab the activity layer of current layer
   for(int b = 0; b < loc->nbatch; b++){
      const pvdata_t * A = getActivity() + b * getNumExtended();
      //We only care about restricted space, but iY and iX are extended
      for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
         for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
            for(int iFeature = 0; iFeature < nf; iFeature++){
              int ext_idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);

              float actualvalue = A[ext_idx];
              
              int xval = (iX + kx0 - loc->halo.lt)/2;
              int yval = (iY + ky0 - loc->halo.up)/2;
              pvErrorIf(!(xval >= 0 && xval < loc->nxGlobal), "Test failed.\n");
              pvErrorIf(!(yval >= 0 && yval < loc->nxGlobal), "Test failed.\n");

              float expectedvalue;
              expectedvalue = iFeature * 64 + yval * 16 + xval * 2 + 4.5;
              expectedvalue*=4;

              if(fabs(actualvalue - expectedvalue) >= 1e-4){
                   pvErrorNoExit() << "Connection " << name << " Mismatch at (" << iX << "," << iY << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".  Discrepancy is a whopping " << actualvalue - expectedvalue << "!  Horrors!" << "\n";
                   isCorrect = false;
              }
            }
         }
      }
   }
   if(!isCorrect){
      Communicator * icComm = parent->getCommunicator();
      MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the run before process 0 reports the error.
      exit(-1);
   }
   return PV_SUCCESS;
}

} /* namespace PV */
