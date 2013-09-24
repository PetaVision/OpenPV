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

FeedbackConn::FeedbackConn(const char * name, HyPerCol * hc, const char * feedforwardConnName) {
    initialize_base();
    initialize(name, hc, feedforwardConnName);
}  // end of FeedbackConn::FeedbackConn(const char *, HyPerCol *, int, GenerativeConn *)

int FeedbackConn::initialize_base() {
   return PV_SUCCESS;
}

int FeedbackConn::initialize(const char * name, HyPerCol *hc, const char * feedforwardConnName) {
   int status = PV_SUCCESS;
   if (hc->parameters()->stringPresent(name, "preLayerName") || hc->parameters()->stringPresent(name, "postLayerName")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\": FeedbackConn does not use preLayerName or postLayerName.\n", hc->parameters()->groupKeywordFromName(name), name);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(hc->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);

   TransposeConn::initialize(name, hc, NULL, NULL, feedforwardConnName);
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

PVPatch *** FeedbackConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);

    for(int arborId = 0; arborId < numAxonalArborLists; arborId++){
       transposeKernels(arborId);
    }
    return arbors;
}  // end of FeedbackConn::initializeWeights

}  // end of namespace PV block

