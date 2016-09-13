#include "AvgPoolTestLayer.hpp"

namespace PV {

AvgPoolTestLayer::AvgPoolTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int AvgPoolTestLayer::updateState(double timef, double dt){
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
   for(int b = 0; b < parent->getNBatch(); b++){
      const pvdata_t * A = getActivity() + b * getNumExtended();
      //We only care about restricted space, but iY and iX are extended
      for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
         for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
            for(int iFeature = 0; iFeature < nf; iFeature++){
              int ext_idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);

              float actualvalue = A[ext_idx];
              
              int xval = iX + kx0 - loc->halo.lt;
              int yval = iY + ky0 - loc->halo.up;
              pvErrorIf(!(xval >= 0 && xval < loc->nxGlobal), "Test failed.\n");
              pvErrorIf(!(yval >= 0 && yval < loc->nxGlobal), "Test failed.\n");

              float expectedvalue;
              if(nxScale == 0.5f){
                 expectedvalue = iFeature * 64 + yval * 16 + xval * 2 + 4.5f;
              }
              else{
                 int res_idx = kIndex(xval, yval, 0, nxGlobal, nyGlobal, 1);
                 //TODO different features define different offsets into this index
                 expectedvalue = iFeature * nxGlobal * nyGlobal + res_idx;
              }
              if(fabsf(actualvalue - expectedvalue) >= 1e-4f){
                   pvErrorNoExit() << "Connection " << name << " Mismatch at (" << iX << "," << iY << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".  Discrepancy is a whopping " << actualvalue - expectedvalue << "!  Horrors!" << "\n";
                   isCorrect = false;
              }
            }
         }
      }
      if(!isCorrect){
         Communicator * icComm = parent->getCommunicator();
         MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the run before process 0 reports the error.
         exit(-1);
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
