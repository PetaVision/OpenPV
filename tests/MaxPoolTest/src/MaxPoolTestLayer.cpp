#include "MaxPoolTestLayer.hpp"

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int MaxPoolTestLayer::updateState(double timef, double dt){
   //Do update state of ANNLayer first
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
   for(int b = 0; b < loc->nbatch; b++){
      //Grab the activity layer of current layer
      const pvdata_t * A = getActivity() + b * getNumExtended();
      //We only care about restricted space
      for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
         for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
            for(int iFeature = 0; iFeature < nf; iFeature++){
              int idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
              //Input image is set up to have max values in 3rd feature dimension
              //3rd dimension, top left is 128, bottom right is 191
              //Y axis spins fastest
              float actualvalue = A[idx];
              
              int xval = iX+kx0-loc->halo.lt;
              int yval = iY+ky0-loc->halo.up;
              //Patches on edges have same answer as previous neuron
              if(xval == 7){
                xval -= 1;
              }
              if(yval == 7){
                     yval -= 1;
              }
              
              // modified GTK: 1/10/15, modified to test spatial max pooling over a feature plane 
              //float expectedvalue = 8*xval+yval+137;
              float expectedvalue = (yval+1)+8*(xval+1)+64*iFeature;
              if(actualvalue != expectedvalue){
                   std::cout << "Connection " << name << " Mismatch at (" << iX << "," << iY << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".  Discrepancy is a whopping " << actualvalue - expectedvalue << "!  Horrors!" << "\n";
                   isCorrect = false;
              }
            }
         }
      }
   }
   if(!isCorrect){
      exit(-1);
   }
   return PV_SUCCESS;
}

BaseObject * createMaxPoolTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new MaxPoolTestLayer(name, hc) : NULL;
}  

} /* namespace PV */
