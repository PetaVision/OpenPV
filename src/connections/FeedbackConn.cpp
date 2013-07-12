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
   char * pre_layer_name;
   char * post_layer_name;
   int status = getPreAndPostLayerNames(feedforwardConnName, hc->parameters(), &post_layer_name, &pre_layer_name);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "FeedbackConn \"%s\" error in rank %d process: unable to get pre- and post-synaptic layer names from originalConnName \"%s\".\n",
            name, hc->columnId(), feedforwardConnName);
      exit(EXIT_SUCCESS);
   }
   assert(pre_layer_name!=NULL && post_layer_name!=NULL);

   TransposeConn::initialize(name, hc, pre_layer_name, post_layer_name, feedforwardConnName);
   free(pre_layer_name);
   free(post_layer_name);
   return PV_SUCCESS;
}

PVPatch *** FeedbackConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);

    transposeKernels();
    return arbors;
}  // end of FeedbackConn::initializeWeights

}  // end of namespace PV block

