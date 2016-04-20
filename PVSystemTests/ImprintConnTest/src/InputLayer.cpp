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

   assert(nf == 4);
   assert(loc->nxGlobal == 2 && loc->nyGlobal == 2);

   //We only care about restricted space
   for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
      for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
         for(int iF = 0; iF < loc->nf; iF++){
            int idx = kIndex(iX, iY, iF, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
            int xval = iX+kx0-loc->halo.lt;
            int yval = iY+ky0-loc->halo.up;

            if(timef == 10 && xval == 0 && yval == 0 && iF == 0){
               A[idx] = 1;
            }
            else if(timef == 10 && xval == 1 && yval == 0 && iF == 1){
               A[idx] = 1;
            }
            else if(timef == 10 && xval == 0 && yval == 1 && iF == 2){
               A[idx] = 1;
            }
            else if(timef == 10 && xval == 1 && yval == 1 && iF == 3){ 
               A[idx] = 1;
            }
            else{
               A[idx] = 0;
            }
         }
      }
   }

   return PV_SUCCESS;
}

BaseObject * createInputLayer(char const * name, HyPerCol * hc) {
   return hc ? new InputLayer(name, hc) : NULL;
}

} /* namespace PV */
