/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer()
{
   initialize_base();
}

ConstantLayer::ConstantLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

ConstantLayer::~ConstantLayer()
{
}

int ConstantLayer::initialize_base()
{
   return PV_SUCCESS;
}

int ConstantLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = ANNLayer::initialize(name, hc);
   return status;
}

void ConstantLayer::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   //This layer is a never a trigger layer, so set flag to false
   if (ioFlag==PARAMS_IO_READ) {
      triggerFlag = 0;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", false);
   }
}

int ConstantLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   //Set the triggerLayer needed by HyPerLayer::needUpdate()
   return status;
}

//Done in HyPerLayer now
//int ConstantLayer::recvAllSynapticInput(){
//   int status = PV_SUCCESS;
//   if (checkIfUpdateNeeded()){
//      status = ANNLayer::recvAllSynapticInput();
//      // doUpdateState will also need to check movieLayer->getLastUpdateTime() against lastUpdateTime,
//      // so wait until then to update lastUpdateTime.
//   }
//   return status;
//}

//int ConstantLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
//      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
//      unsigned int * active_indices, unsigned int * num_active)
//{
//   update_timer->start();
//   int status = PV_SUCCESS;
//   if (checkIfUpdateNeeded()){
//      status = ANNLayer::doUpdateState(time,  dt, loc, A, V, num_channels, gSynHead, spiking, active_indices, num_active);
//      lastUpdateTime = parent->simulationTime();
//   }
//   update_timer->stop();
//   return status;
//}

//bool ConstantLayer::checkIfUpdateNeeded() {
bool ConstantLayer::needUpdate(double time, double dt) {
   //Only update on initialization
   assert(time >= parent->getStartTime());
   if (time == parent->getStartTime()){
       return true;
   }
   else{
       return false;
   }
}

BaseObject * createConstantLayer(char const * name, HyPerCol * hc) {
   return hc ? new ConstantLayer(name, hc) : NULL;
}

} /* namespace PV */

