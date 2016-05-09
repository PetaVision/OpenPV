#include "BinningTestLayer.hpp"

namespace PV {

BinningTestLayer::BinningTestLayer(const char * name, HyPerCol * hc){
   BinningLayer::initialize(name, hc);
}

int BinningTestLayer::updateState(double timef, double dt){
   //Do update state of Binning Layer first
   BinningLayer::updateState(timef, dt);
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;
   //Grab the activity layer of current layer
   const pvdata_t * A = getActivity();
   //We only care about restricted space
   for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
      for(int iX = loc->halo.up; iX < nx + loc->halo.lt; iX++){
         for(int iF = 0; iF < nf; iF++){
            int idx = kIndex(iX, iY, iF, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
            if(getSigma() == 0){
               //Based on the input image, X index should be the same as F index
               if(iX+kx0 == iF){
                  if(A[idx] != 1){
                     std::cout << iY << "," << iX << "," << iF << ": " << A[idx] << "\n";
                  }
                  assert(A[idx] == 1);
               }
               else{
                  if(A[idx] != 0){
                     std::cout << iY << "," << iX << "," << iF << ": " << A[idx] << "\n";
                  }
                  assert(A[idx] == 0);
               }
            }
            else if (getSigma() == 2){
               float temp;
               if(iX+kx0 == iF-2 || iX+kx0 == iF+2){
                  temp = A[idx] - .121;
                  if(temp > .00001){
                     std::cout << iY << "," << iX << "," << iF << ": " << A[idx] << "\n";
                  }
                  assert(temp <= .00001);
               }
               if(iX+kx0 == iF-1 || iX+kx0 == iF+1){
                  temp = A[idx] - .176;
                  if(temp > .00001){
                     std::cout << iY << "," << iX << "," << iF << ": " << A[idx] << "\n";
                  }
                  assert(temp <= .00001);

               }
               if(iX+kx0 == iF){
                  temp = A[idx] - .199;
                  if(temp > .00001){
                     std::cout << iY << "," << iX << "," << iF << ": " << A[idx] << "\n";
                  }
                  assert(temp <= .00001);
               }
            }
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createBinningTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new BinningTestLayer(name, hc) : NULL;
}

} /* namespace PV */
