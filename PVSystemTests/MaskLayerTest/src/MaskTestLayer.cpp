#include "MaskTestLayer.hpp"

namespace PV {

MaskTestLayer::MaskTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

MaskTestLayer::~MaskTestLayer(){
   if(maskMethod){
      free(maskMethod);
   }
}
int MaskTestLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_maskMethod(ioFlag);
   return status;
}

void MaskTestLayer::ioParam_maskMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "maskMethod", &maskMethod);
   //Check valid methods
   if(strcmp(maskMethod, "layer") == 0){
   }
   else if(strcmp(maskMethod, "maskFeatures") == 0){
   }
   else if(strcmp(maskMethod, "noMaskFeatures") == 0){
   }
   else{
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: \"%s\" is not a valid maskMethod. Options are \"layer\", \"maskFeatures\", or \"noMaskFeatures\".\n",
                 getKeyword(), name, maskMethod);
      }
      exit(-1);
   }
}


int MaskTestLayer::updateState(double timef, double dt){
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;

   bool isCorrect = true;
   for(int b = 0; b < loc->nbatch; b++){
      pvdata_t * GSynExt = getChannel(CHANNEL_EXC) + b * getNumNeurons(); //gated
      pvdata_t * GSynInh = getChannel(CHANNEL_INH) + b * getNumNeurons(); //gt
      pvdata_t * GSynInhB = getChannel(CHANNEL_INHB) + b * getNumNeurons(); //mask

      //Grab the activity layer of current layer
      //We only care about restricted space
      
      for (int k = 0; k < getNumNeurons(); k++){
         if(strcmp(maskMethod, "layer") == 0){
         //std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
            if(GSynInhB[k]){
               if(GSynExt[k] != GSynInh[k]){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
                   isCorrect = false;
               }
            }
            else{
               if(GSynExt[k] != 0){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                   isCorrect = false;
               }
            }
         }
         else if(strcmp(maskMethod, "maskFeatures") == 0){
            int featureIdx = featureIndex(k, nx, ny, nf);
            //Param files specifies idxs 0 and 2 out of 3 total features
            if(featureIdx == 0 || featureIdx == 2){
               if(GSynExt[k] != 0){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
                   isCorrect = false;
               }
            }
            else{
               if(GSynExt[k] != GSynInh[k]){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
                   isCorrect = false;
               }
            }
         }
         else if(strcmp(maskMethod, "noMaskFeatures") == 0){
            int featureIdx = featureIndex(k, nx, ny, nf);
            //Param files specifies idxs 0 and 2 out of 3 total features
            if(featureIdx == 0 || featureIdx == 2){
               if(GSynExt[k] != GSynInh[k]){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
                   isCorrect = false;
               }
            }
            else{
               if(GSynExt[k] != 0){
                   std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: 0.\n";
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

BaseObject * createMaskTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new MaskTestLayer(name, hc) : NULL;
}

} /* namespace PV */
