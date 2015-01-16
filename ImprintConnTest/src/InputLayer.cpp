#include "InputLayer.hpp"

namespace PV {

InputLayer::InputLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int InputLayer::updateState(double timef, double dt){
   //Grab the activity layer of current layer
   pvdata_t * A = getActivity();
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;

   assert(nf == 1);
   assert(loc->nxGlobal == 2 && loc->nyGlobal == 2);

   //We only care about restricted space
   for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
      for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
         int idx = kIndex(iX, iY, 0, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
         if(timef < 5){
            A[idx] = 0;
         }
         else{
            A[idx] = .1;
         }
      }
   }

   return PV_SUCCESS;
}



} /* namespace PV */
