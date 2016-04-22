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
         fprintf(stderr, "%s \"%s\": FeedbackConn does not use preLayerName or postLayerName.\n", this->getKeyword(), name);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
   return status;
}

int FeedbackConn::handleMissingPreAndPostLayerNames() {
   assert(originalConn && originalConn->getInitInfoCommunicatedFlag());
   assert(originalConn->getPreLayerName() && originalConn->getPostLayerName());
   preLayerName = strdup(originalConn->getPostLayerName());
   postLayerName = strdup(originalConn->getPreLayerName());
   if (preLayerName==NULL || postLayerName==NULL) {
      fprintf(stderr, "Error in rank %d process: FeedbackConn \"%s\" unable to allocate memory for pre and post layer names: %s", parent->columnId(), name, strerror(errno));
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

BaseObject * createFeedbackConn(char const * name, HyPerCol * hc) {
   return hc ? new FeedbackConn(name, hc) : NULL;
}

}  // end of namespace PV block

