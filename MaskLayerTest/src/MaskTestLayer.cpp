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

   bool isCorrect = true;
   //Grab the activity layer of current layer
   //We only care about restricted space
   int numActive = 0;
   for (int k = 0; k < getNumNeurons(); k++){
   //std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
      if(GSynExt[k]){
         numActive++;
         if(GSynExt[k] != GSynInh[k]){
             std::cout << "Connection " << name << " Mismatch at " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
             isCorrect = false;
         }
      }
   }
   
   //Make sure all activity isn't 0
   assert(numActive != 0);

   if(!isCorrect){
      exit(-1);
   }
   return PV_SUCCESS;
}



} /* namespace PV */
