#include "ImageOffsetTestLayer.hpp"

namespace PV {

#ifdef PV_USE_GDAL

ImageOffsetTestLayer::ImageOffsetTestLayer(const char * name, HyPerCol * hc){
   Image::initialize(name, hc);
}

double ImageOffsetTestLayer::getDeltaUpdateTime(){
   return 1;
}

int ImageOffsetTestLayer::updateState(double timef, double dt){
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;
   assert(loc->halo.up == 0 && loc->halo.lt==0 && loc->halo.rt == 0 && loc->halo.dn == 0);

   bool isCorrect = true;
   //Grab the activity layer of current layer
   for(int b = 0; b < loc->nbatch; b++){
      const pvdata_t * A = getActivity() + b * getNumExtended();
      //We only care about restricted space
      for(int iY = 0; iY < ny; iY++){
         for(int iX = 0; iX < nx; iX++){
            for(int iF = 0; iF < nf; iF++){
               int idx = kIndex(iX, iY, iF, nx, ny, nf);
               int ixGlobal = kx0 + iX;
               int iyGlobal = ky0 + iY;
               float actualvalue = A[idx]*255;
               float expectedvalue = -1;
               if(strcmp(name, "crop") == 0){
                  expectedvalue = (iyGlobal + 4) * 16 + (ixGlobal + 4);
               }
               else if(strcmp(name, "pad") == 0){
                  if(ixGlobal < 8 || iyGlobal < 8 || ixGlobal >= 24 || iyGlobal >= 24){
                     expectedvalue = 0;
                  }
                  else{
                     expectedvalue = (iyGlobal - 8) * 16 + (ixGlobal - 8);
                  }
               }
               else if(strcmp(name, "TLCorner") == 0){
                  if(ixGlobal < 2 && iyGlobal < 2){
                     expectedvalue = (iyGlobal + 14) * 16 + (ixGlobal + 14);
                  }
                  else{
                     expectedvalue = 0;
                  }
               }
               else if(strcmp(name, "TRCorner") == 0){
                  if(ixGlobal >= 14 && iyGlobal < 2){
                     expectedvalue = (iyGlobal + 14) * 16 + (ixGlobal - 14);
                  }
                  else{
                     expectedvalue = 0;
                  }
               }
               else if(strcmp(name, "BLCorner") == 0){
                  if(ixGlobal < 2 && iyGlobal >= 14){
                     expectedvalue = (iyGlobal - 14) * 16 + (ixGlobal + 14);
                  }
                  else{
                     expectedvalue = 0;
                  }
               }
               else if(strcmp(name, "BRCorner") == 0){
                  if(ixGlobal >= 14 && iyGlobal >= 14){
                     expectedvalue = (iyGlobal - 14) * 16 + (ixGlobal - 14);
                  }
                  else{
                     expectedvalue = 0;
                  }
               }
               else if(strcmp(name, "TLOver") == 0){
                  if(ixGlobal >= 14 || iyGlobal >= 14){
                     expectedvalue = 0;
                  }
                  else{
                     expectedvalue = (iyGlobal + 2) * 16 + (ixGlobal + 2);
                  }
               }
               else if(strcmp(name, "TROver") == 0){
                  if(ixGlobal < 2 || iyGlobal >= 14){
                     expectedvalue = 0;
                  }
                  else{
                     expectedvalue = (iyGlobal + 2) * 16 + (ixGlobal - 2);
                  }
               }
               else if(strcmp(name, "BLOver") == 0){
                  if(ixGlobal >= 14 || iyGlobal < 2){
                     expectedvalue = 0;
                  }
                  else{
                     expectedvalue = (iyGlobal - 2) * 16 + (ixGlobal + 2);
                  }
               }
               else if(strcmp(name, "BROver") == 0){
                  if(ixGlobal < 2 || iyGlobal < 2){
                     expectedvalue = 0;
                  }
                  else{
                     expectedvalue = (iyGlobal - 2) * 16 + (ixGlobal - 2);
                  }
               }
               else{
                  std::cout << "Layer name " << name << " not recoginzed\n";
                  exit(-1);
               }
               float diff = fabs(actualvalue - expectedvalue);
               if(diff >= 1e-4){
                  std::cout << "Connection " << name << " Mismatch at (" << ixGlobal << "," << iyGlobal << "," << iF << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << "\n";
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

#else // PV_USE_GDAL
ImageOffsetTestLayer::ImageOffsetTestLayer(const char * name, HyPerCol * hc) {
   if (hc->columnId()==0) {
      fprintf(stderr, "ImageOffsetTestLayer class requires compiling with PV_USE_GDAL set\n");
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   exit(EXIT_FAILURE);
}
#endif // PV_USE_GDAL

BaseObject * createImageOffsetTestLayer(const char * name, HyPerCol * hc) {
   return hc ? new ImageOffsetTestLayer(name, hc) : NULL;
}

} /* namespace PV */
