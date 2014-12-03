/*
 * WindowConn.cpp
 *
 *  Created on: Nov 25, 2014
 *      Author: pschultz
 */

#ifdef OBSOLETE // Marked obsolete Dec 2, 2014.  Use sharedWeights=false instead of windowing.

#include "WindowConn.hpp"

namespace PV {

WindowConn::WindowConn(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

WindowConn::WindowConn() {
   initialize_base();
}

int WindowConn::initialize_base() {
   return PV_SUCCESS;
}

int WindowConn::initialize(char const * name, HyPerCol * hc) {
   return HyPerConn::initialize(name, hc);
}

void WindowConn::ioParam_useWindowPost(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "sharedWeights"));
   assert(!parent->parameters()->presentAndNotBeenRead(name, "numAxonalArbors"));
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if (sharedWeights && plasticityFlag && numAxonalArborLists>1) {
      initialWeightUpdateTime = 1.0;
      parent->ioParamValue(ioFlag, name, "useWindowPost", &useWindowPost, useWindowPost);
   }
}

int WindowConn::defaultUpdateInd_dW(int arbor_ID, int kExt){
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();

   bool inWindow = true;
   // only check inWindow if number of arbors > 1
   if (this->numberOfAxonalArborLists()>1){
      int kPost = layerIndexExt(kExt, preLoc, postLoc);
      inWindow = post->inWindowExt(arbor_ID, kPost);
      if(useWindowPost){
      }
      else{
         inWindow = pre->inWindowExt(arbor_ID, kExt);
      }
      if(!inWindow) return PV_CONTINUE;
   }

   return HyPerConn::defaultUpdateInd_dW(arbor_ID, kExt);
}

WindowConn::~WindowConn() {
}

} /* namespace PV */

#endif // OBSOLETE
