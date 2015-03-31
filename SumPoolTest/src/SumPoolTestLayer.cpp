#include "SumPoolTestLayer.hpp"

namespace PV {

SumPoolTestLayer::SumPoolTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int SumPoolTestLayer::updateState(double timef, double dt){
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
   const pvdata_t * A = getActivity();
   //We only care about restricted space, but iY and iX are extended
   for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
      for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
         for(int iFeature = 0; iFeature < nf; iFeature++){
           int ext_idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);

           float actualvalue = A[ext_idx];
           
           int xval = iX + kx0 - loc->halo.lt;
           int yval = iY + ky0 - loc->halo.up;
           assert(xval >= 0 && xval < loc->nxGlobal);
           assert(yval >= 0 && yval < loc->nxGlobal);

           int res_idx = kIndex(xval, yval, 0, nxGlobal, nyGlobal, 1);
           //TODO different features define different offsets into this index
           
           float expectedvalue = iFeature * nxGlobal * nyGlobal + res_idx;
           if(fabs(actualvalue - expectedvalue) >= 1e-4){
                std::cout << "Connection " << name << " Mismatch at (" << iX << "," << iY << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".  Discrepancy is a whopping " << actualvalue - expectedvalue << "!  Horrors!" << "\n";
                isCorrect = false;
           }
         }
      }
   }
   if(!isCorrect){
      exit(-1);
   }
   return PV_SUCCESS;
}



} /* namespace PV */
