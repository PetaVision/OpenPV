#include "ProbeLayer.hpp"

namespace PV {

ProbeLayer::ProbeLayer(const char * name, HyPerCol * hc){
   ANNLayer::initialize(name, hc);
   sumDistance = 0;
   numExamples = 0;
   //TODO make this variable a parameter
   dispPeriod = 128;
}

int ProbeLayer::updateState(double timef, double dt){
   //Do update state of ANN Layer first
   ANNLayer::updateState(timef, dt);
   
   pvdata_t * GSynExt = getChannel(CHANNEL_EXC); //gt
   pvdata_t * GSynInh = getChannel(CHANNEL_INH); //guess

   const PVLayerLoc * loc = getLayerLoc(); 
   double sumCost = 0;
   double maxGuessVal = -999999;
   double maxGTVal = -999999;
   int maxGuessIdx = -1;
   int maxGTIdx = -1;
   //Calculate cost function
   for(int ni = 0; ni < getNumNeurons(); ni++){
      if(GSynExt[ni] == 1){
         sumCost += log(GSynInh[ni]);
      }
      if(GSynExt[ni] >= maxGuessVal){
         maxGuessVal = GSynExt[ni];
         maxGuessIdx = ni;
      }
      if(GSynInh[ni] >= maxGTVal){
         maxGTVal = GSynInh[ni];
         maxGTIdx = ni;
      }
   }
   assert(maxGuessIdx >= 0);
   assert(maxGTIdx >= 0);

   sumDistance += -sumCost;
   numCorrect += maxGuessIdx == maxGTIdx ? 1 : 0;
   numExamples++;

   if(numExamples % dispPeriod != 0) return PV_SUCCESS;
   std::cout << "Time: " << timef << " Current distance: " << (-sumCost) << "  Average distance: " << sumDistance/numExamples << "  Average Score: " << (float)numCorrect / numExamples << "\n";

   std::cout << "Guess: [";
   for(int ni = 0; ni < getNumNeurons(); ni++){
      std::cout << GSynInh[ni] << ",";
   }
   std::cout << "] \n Actual: [";
   for(int ni = 0; ni < getNumNeurons(); ni++){
      std::cout << GSynExt[ni] << ",";
   }
   std::cout << "]\n";

   numExamples = 0;
   sumDistance = 0;
   numCorrect = 0;


   return PV_SUCCESS;
}



} /* namespace PV */
