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
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;
   assert(nf == 3);

   bool isCorrect = true;
   //Grab the activity layer of current layer
   const pvdata_t * A = getActivity();
   //We only care about restricted space
   for(int iY = loc->halo.up+1; iY < ny + loc->halo.up-1; iY++){
      for(int iX = loc->halo.lt+1; iX < nx + loc->halo.lt-1; iX++){
	for(int iFeature = 0; iFeature < nf; iFeature++){
	  int idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
	  int restricted_idx = kIndexRestricted(idx, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
	  //3rd dimension, top left is 128, bottom right is 191
	  //Y axis spins fastest
	  float actualvalue = A[idx];
	  
	  int xval = iX+kx0-loc->halo.lt;
	  int yval = iY+ky0-loc->halo.up;
	  
	  //expectedvalue = restricted index (except on edges)
	  float expectedvalue = restricted_idx;
	  if(actualvalue != expectedvalue){
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
