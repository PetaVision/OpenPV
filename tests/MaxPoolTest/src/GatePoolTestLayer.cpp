#include "GatePoolTestLayer.hpp"

namespace PV {

GatePoolTestLayer::GatePoolTestLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
}

int GatePoolTestLayer::updateState(double timef, double dt){

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

      //Grab the activity layer of current layer
      //We only care about restricted space
      int numActive = 0;
      for (int k = 0; k < getNumNeurons(); k++){
         if(GSynExt[k]){
            numActive++;
            if(GSynExt[k] != GSynInh[k]){
                std::cout << "Connection " << name << " Mismatch at batch " << b << " neuron " << k << ": actual value: " << GSynExt[k] << " Expected value: " << GSynInh[k] << ".\n";
                isCorrect = false;
            }
         }
      }
      
      //Must be 25% active
      float percentActive = (float)numActive/getNumNeurons();
      if(percentActive != .25){
         std::cout << "Percent active for " << name << " is " << percentActive << ", where expected is .25 at timestep " << timef << " for batch " << b << "\n";
      }
      assert(percentActive == .25);
   }

   if(!isCorrect){
      exit(-1);
   }
   return PV_SUCCESS;
}

BaseObject * createGatePoolTestLayer(char const * name, HyPerCol * hc) {
   return hc ? new GatePoolTestLayer(name, hc) : NULL;
}

} /* namespace PV */
