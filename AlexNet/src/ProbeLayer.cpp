#include "ProbeLayer.hpp"

namespace PV {

ProbeLayer::ProbeLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int ProbeLayer::updateState(double timef, double dt){
   //Do update state of ANN Layer first
   ANNLayer::updateState(timef, dt);

   pvdata_t * GSynExt = getChannel(CHANNEL_EXC); //gt
   pvdata_t * GSynInh = getChannel(CHANNEL_INH); //guess

   const PVLayerLoc * loc = getLayerLoc(); 
   float thresh = .5;
   for(int ni = 0; ni < getNumNeurons(); ni++){
      float guess = GSynExt[ni] <= thresh ? 0:1;
      float actual = GSynInh[ni];
      assert(guess == actual);
      //std::cout << "guess: " << guess << " actual: " << actual << "\n";

   }
   return PV_SUCCESS;
}



} /* namespace PV */
