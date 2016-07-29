/*
 * FeedbackConn.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: pschultz
 */

#include "FeedbackConn.hpp"

namespace PV {

FeedbackConn::FeedbackConn() {
    initialize_base();
}

FeedbackConn::FeedbackConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int FeedbackConn::initialize_base() {
   return PV_SUCCESS;
}

int FeedbackConn::initialize(const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   return TransposeConn::initialize(name, hc);
}

// FeedbackConn doesn't use preLayerName or postLayerName
// If they're present, errors are handled byy setPreAndPostLayerNames
void FeedbackConn::ioParam_preLayerName(enum ParamsIOFlag ioFlag) {}
void FeedbackConn::ioParam_postLayerName(enum ParamsIOFlag ioFlag) {}

int FeedbackConn::setPreAndPostLayerNames() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   if (params->stringPresent(name, "preLayerName") || params->stringPresent(name, "postLayerName")) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s: FeedbackConn does not use preLayerName or postLayerName.\n", getDescription_c());
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
   return status;
}

int FeedbackConn::handleMissingPreAndPostLayerNames() {
   assert(originalConn && originalConn->getInitInfoCommunicatedFlag());
   assert(originalConn->getPreLayerName() && originalConn->getPostLayerName());
   preLayerName = strdup(originalConn->getPostLayerName());
   postLayerName = strdup(originalConn->getPreLayerName());
   if (preLayerName==NULL || postLayerName==NULL) {
      pvError().printf("%s: Rank %d process unable to allocate memory for pre and post layer names: %s", getDescription_c(), parent->columnId(), strerror(errno));
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block

