#include "TestLayer.hpp"

namespace PV {

TestLayer::TestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int TestLayer::updateState(double timef, double dt){
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
   assert(loc->nxGlobal == 8);
   assert(loc->nyGlobal == 8);

   bool isCorrect = true;
   //Grab the activity layer of current layer
   const pvdata_t * A = getActivity();
   //We only care about restricted space
   for(int iY = loc->halo.up; iY < ny + loc->halo.up; iY++){
      for(int iX = loc->halo.lt; iX < nx + loc->halo.lt; iX++){
         for(int iFeature = 0; iFeature < nf; iFeature++){
            int idx = kIndex(iX, iY, iFeature, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
            //3rd dimension, top left is 128, bottom right is 191
            //Y axis spins fastest
            float actualvalue = A[idx];

            int xval = iX+kx0-loc->halo.lt;
            int yval = iY+ky0-loc->halo.up;

            //Averaged across all 4 patches. Calculate averages first
            //Figure out location in patch size
            int xPatchLoc = xval % 4;
            int yPatchLoc = yval % 4;
            int xPatchIdx = xval/4;
            int yPatchIdx = yval/4;
            //Value should be average of same spot in all 4 patches
            float expectedvalue;
            if(timef == 10){
               expectedvalue = (iFeature*64 + xPatchLoc * 8 + yPatchLoc) + 
                                     (iFeature*64 + (xPatchLoc+4) * 8 + yPatchLoc) +
                                     (iFeature*64 + xPatchLoc * 8 + (yPatchLoc+4)) +
                                     (iFeature*64 + (xPatchLoc+4) * 8 + (yPatchLoc+4));
               expectedvalue = expectedvalue / 4;

               //expectedvalue should also be dependent on which kernel is active
               //Order of imprinting: kernel 0 at ts 4, kernel 1 at ts 5, kernel 2 at ts 6, kernel 3 at ts 7
               //Values of imprinting should be orig value * (ts + 1)
               int tsScale;
               if(     xPatchIdx == 0 && yPatchIdx == 0){
                  tsScale = 5;
               }
               else if(xPatchIdx == 1 && yPatchIdx == 0){
                  tsScale = 6;
               }
               else if(xPatchIdx == 0 && yPatchIdx == 1){
                  tsScale = 7;
               }
               else if(xPatchIdx == 1 && yPatchIdx == 1){
                  tsScale = 8;
               }
               expectedvalue = expectedvalue * tsScale;
            }
            else{
               expectedvalue = 0;
            }

            if(actualvalue != expectedvalue){
               std::cout << "Connection " << name << " Mismatch at (" << xval << "," << yval << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".\n";
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

BaseObject * createTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new TestLayer(name, hc) : NULL;
}

} /* namespace PV */
