/*
 * BaseLayer.cpp
 *
 *  Created on: Jan 16, 2010
 *      Author: Craig Rasmussen
 */

#include "BaseLayer.hpp"

namespace PV {

BaseLayer::BaseLayer() {
//   initialize_base();
}

//int BaseLayer::initialize_base() {
//   dataType = PV_FLOAT;
//   dataTypeString = NULL;
//   this->initInfoCommunicatedFlag = false;
//   this->dataStructuresAllocatedFlag = false;
//   this->initialValuesSetFlag = false;
//   this->recvGpu = false;
//   this->updateGpu = false;
//   phase = 0;
//   sparseLayer = false;
//   this->clayer = NULL;
//   this->lastUpdateTime = 0.0;
//}

BaseLayer::~BaseLayer()
{
//   if(dataTypeString){
//      free(dataTypeString);
//   }
}

//int BaseLayer::ioParams(enum ParamsIOFlag ioFlag)
//{
//   //Moved to BaseLayer
//   parent->ioParamsStartGroup(ioFlag, name);
//   ioParamsFillGroup(ioFlag);
//   parent->ioParamsFinishGroup(ioFlag);
//
//   return PV_SUCCESS;
//}
//
//int BaseLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
//   ioParam_dataType(ioFlag);
//
//}
//
//void BaseLayer::ioParam_dataType(enum ParamsIOFlag ioFlag) {
//   this->getParent()->ioParamString(ioFlag, this->getName(), "dataType", &dataTypeString, NULL, false/*warnIfAbsent*/);
//   if(dataTypeString == NULL){
//      //Default value
//      dataType = PV_FLOAT;
//      return;
//   }
//   if(!strcmp(dataTypeString, "float")){
//      dataType = PV_FLOAT;
//   }
//   else if(!strcmp(dataTypeString, "int")){
//      dataType = PV_INT;
//   }
//   else{
//      std::cout << "BaseLayer " << name << " Error: dataType not recognized, can be \"float\" or \"int\"\n";
//      exit(-1);
//   }
//}
//
//int BaseLayer::initialize(const char * name, HyPerCol * hc) {
//   this->name = strdup(name);
//   setParent(hc); // Could this line and the parent->addLayer line be combined in a HyPerLayer method?
//
//   PVParams * params = parent->parameters();
//   int status = ioParams(PARAMS_IO_READ);
//   assert(status == PV_SUCCESS);
//
//   layerId = parent->addLayer(this);
//}

} // namespace PV


