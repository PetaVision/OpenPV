/*
 * GeislerLayer.cpp
 *
 *  Created on: Apr 21, 2010
 *      Author: gkenyon
 */

#include "ODDLayer.hpp"
#include <assert.h>
#include <float.h>

namespace PV {

ODDLayer::ODDLayer() {
   initialize_base();
}

ODDLayer::ODDLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ODDLayer::ODDLayer(const char * name, HyPerCol * hc, PVLayerType type) {
   initialize_base();
   initialize(name, hc);
}

ODDLayer::~ODDLayer() {
}

int ODDLayer::initialize_base() {
   return PV_SUCCESS;
}

int ODDLayer::initialize(const char * name, HyPerCol * hc) {
   return ANNLayer::initialize(name, hc, MAX_CHANNELS);
}

int ODDLayer::updateState(float time, float dt)
{

   pv_debug_info("[%d]: ODDLayer::updateState:", clayer->columnId);

   pvdata_t * V = clayer->V;
   pvdata_t * phiExc  = getChannel(CHANNEL_EXC);
   pvdata_t * phiInh  = getChannel(CHANNEL_INH);


   // assume bottomUp input to phiExc, lateral input to phiInh
   for (int k = 0; k < clayer->numNeurons; k++) {
      pvdata_t bottomUp_input = phiExc[k];
      pvdata_t lateral_input = phiInh[k];
      V[k] = (bottomUp_input > 0.0f) ? bottomUp_input * lateral_input : bottomUp_input;
   }

   resetGSynBuffers();
   applyVMax();
   applyVThresh();
   setActivity();


   return 0;
}

}
