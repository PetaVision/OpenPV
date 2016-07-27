/*
 * BaseObject.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "BaseObject.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

BaseObject::BaseObject() {
   initialize_base();
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call BaseObject::initialize in their own
   // constructor.
}

int BaseObject::initialize_base() {
   name = NULL;
   parent = NULL;
   return PV_SUCCESS;
}

int BaseObject::initialize(const char * name, HyPerCol * hc) {
   int status = setName(name);
   if (status==PV_SUCCESS) { status = setParent(hc); }
   if (status==PV_SUCCESS) { status = setDescription(); }
   return status;
}

char const * BaseObject::getKeyword() const {
   return getParent()->parameters()->groupKeywordFromName(getName());
}

int BaseObject::setName(char const * name) {
   pvAssert(this->name==NULL);
   int status = PV_SUCCESS;
   this->name = strdup(name);
   if (this->name==NULL) {
      pvErrorNoExit().printf("could not set name \"%s\": %s\n", name, strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}

int BaseObject::setParent(HyPerCol * hc) {
   pvAssert(parent==NULL);
   HyPerCol * parentCol = dynamic_cast<HyPerCol*>(hc);
   int status = parentCol!=NULL ? PV_SUCCESS : PV_FAILURE;
   if (parentCol) {
      parent = parentCol;
   }
   return status;
}

int BaseObject::setDescription() {
   description.clear();
   description.append(getKeyword()).append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int BaseObject::respond(std::shared_ptr<BaseMessage> message) {
   int status = PV_SUCCESS; // TODO: convert PV_SUCCESS, PV_FAILURE, etc. to enum
   if (message==nullptr) {
      return PV_SUCCESS;
   }
   else if (CommunicateInitInfoMessage<BaseObject*> const * castMessage = dynamic_cast<CommunicateInitInfoMessage<BaseObject*> const*>(message.get())) {
      status = respondCommunicateInitInfo(castMessage);
   }
   else if (AllocateDataMessage const * castMessage = dynamic_cast<AllocateDataMessage const*>(message.get())) {
      status = respondAllocateData(castMessage);
   }
   else if (InitializeStateMessage const * castMessage = dynamic_cast<InitializeStateMessage const*>(message.get())) {
      status = respondInitializeState(castMessage);
   }
   else if (ConnectionUpdateMessage const * castMessage = dynamic_cast<ConnectionUpdateMessage const*>(message.get())) {
      status = respondConnectionUpdate(castMessage);
   }
   else if (ConnectionFinalizeUpdateMessage const * castMessage = dynamic_cast<ConnectionFinalizeUpdateMessage const*>(message.get())) {
      status = respondConnectionFinalizeUpdate(castMessage);
   }
   else if (ConnectionOutputMessage const * castMessage = dynamic_cast<ConnectionOutputMessage const*>(message.get())) {
      status = respondConnectionOutput(castMessage);
   }
   else if (LayerUpdateStateMessage const * castMessage = dynamic_cast<LayerUpdateStateMessage const*>(message.get())) {
      status = respondLayerUpdateState(castMessage);
   }
   else if (LayerRecvSynapticInputMessage const * castMessage = dynamic_cast<LayerRecvSynapticInputMessage const*>(message.get())) {
      status = respondLayerRecvSynapticInput(castMessage);
   }
#ifdef PV_USE_CUDA
   else if (LayerCopyFromGpuMessage const * castMessage = dynamic_cast<LayerCopyFromGpuMessage const*>(message.get())) {
      status = respondLayerCopyFromGpu(castMessage);
   }
#endif // PV_USE_CUDA
   else if (LayerPublishMessage const * castMessage = dynamic_cast<LayerPublishMessage const*>(message.get())) {
      status = respondLayerPublish(castMessage);
   }
   else if (LayerUpdateActiveIndicesMessage const * castMessage = dynamic_cast<LayerUpdateActiveIndicesMessage const*>(message.get())) {
      status = respondLayerUpdateActiveIndices(castMessage);
   }
   else if (LayerOutputStateMessage const * castMessage = dynamic_cast<LayerOutputStateMessage const*>(message.get())) {
      status = respondLayerOutputState(castMessage);
   }
   else if (LayerCheckNotANumberMessage const * castMessage = dynamic_cast<LayerCheckNotANumberMessage const*>(message.get())) {
      status = respondLayerCheckNotANumber(castMessage);
   }
   else {
      pvError() << "Unrecognized message type\n";
      status = PV_FAILURE;
   }
   return status;
}

BaseObject::~BaseObject() {
   free(name);
}

} /* namespace PV */
