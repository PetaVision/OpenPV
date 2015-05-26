#include "MaskTestLayer.hpp"

namespace PV {

MaskTestLayer::MaskTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int MaskTestLayer::updateState(double timef, double dt){
   //Grab layer size
   const PVLayerLoc* loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int kx0 = loc->kx0;
   int ky0 = loc->ky0;

   pvdata_t * GSynExt = getChannel(CHANNEL_EXC); //gated
   pvdata_t * GSynInh = getChannel(CHANNEL_INH); //gt
   pvdata_t * GSynInhB = getChannel(CHANNEL_INHB); //mask

   bool isCorrect = true;
   //Grab the activity layer of current layer
   //We only care about restricted space
   
   for (int k = 0; k < getNumNeurons(); k++){
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
   

   if(!isCorrect){
      exit(-1);
   }
   return PV_SUCCESS;
}



} /* namespace PV */
