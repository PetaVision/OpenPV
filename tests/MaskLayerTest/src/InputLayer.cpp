#include "InputLayer.hpp"

namespace PV {

InputLayer::InputLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

//Makes a layer such that the restricted space is the index, but with spinning order be [x, y, f] as opposed to [f, x, y]
int InputLayer::updateState(double timef, double dt){
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;

   pvdata_t * A = getActivity();
   //looping over ext
   for(int iY = 0; iY < ny+loc->halo.up+loc->halo.dn; iY++){
      for(int iX = 0; iX < nx+loc->halo.lt+loc->halo.rt; iX++){
         //Calculate x and y global extended
         int xGlobalExt = iX + loc->kx0;
         int yGlobalExt = iY + loc->ky0;
         //Calculate x and y in restricted space
         int xGlobalRes = xGlobalExt - loc->halo.lt;
         int yGlobalRes = yGlobalExt - loc->halo.up;
         //Calculate base value
         //xGlobal and yGlobalRes can be negative
         int baseActivityVal = yGlobalRes * nxGlobal + xGlobalRes;

         for(int iFeature = 0; iFeature < nf; iFeature++){
           int ext_idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
           //Feature gives an offset, since it spins slowest
           int activityVal = baseActivityVal + iFeature * nxGlobal * nyGlobal;
           A[ext_idx] = activityVal;
         }
      }
   }

   ////Printing for double checking 
   //printf("\nOutMat\n");
   ////looping over ext
   //for(int iFeature = 0; iFeature < nf; iFeature++){
   //   for(int iY = 0; iY < ny+loc->halo.up+loc->halo.dn; iY++){
   //      for(int iX = 0; iX < nx+loc->halo.lt+loc->halo.rt; iX++){
   //         int ext_idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
   //         printf("%03d ", (int)A[ext_idx]);
   //      }
   //      printf("\n");
   //   }
   //   printf("\n\n");
   //}

   return PV_SUCCESS;
}

BaseObject * createInputLayer(char const * name, HyPerCol * hc) {
   return hc ? new InputLayer(name, hc) : NULL;
}

} /* namespace PV */
