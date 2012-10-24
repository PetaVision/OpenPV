/*
 * OnlineLearningKConn.cpp
 *
 *  Created on: Sep 12, 2012
 *      Author: pschultz
 */

#include "OnlineLearningKConn.hpp"

namespace PV {

OnlineLearningKConn::OnlineLearningKConn() {
   initialize_base();
}

OnlineLearningKConn::OnlineLearningKConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                    const char * filename, InitWeights * weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
}


OnlineLearningKConn::~OnlineLearningKConn() {
}

int OnlineLearningKConn::initialize_base() {
   sourceLayer = NULL;
   postpostOuterProduct = NULL;
   prepostOuterProduct = NULL;

   return PV_SUCCESS;
}

int OnlineLearningKConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights * weightInit) {
   int status = KernelConn::initialize(name, hc, pre, post, filename, weightInit);

   PVParams * params = getParent()->parameters(); // parent was during call to KernelConn::initialize
   const char * sourceLayerName = params->stringValue(name, "sourceLayer");
   if (sourceLayerName == NULL) {
      if (getParent()->columnId() == 0) {
         fprintf(stderr, "OnlineLearningKConn \"%s\": params file must specify sourceLayer\n", getName());
      }
      abort();
   }

   sourceLayer = getParent()->getLayerFromName(sourceLayerName);
   if (sourceLayer==NULL) {
      if (getParent()->columnId() == 0) {
         fprintf(stderr, "OnlineLearningKConn \"%s\": sourceLayer \"%s\" is not a defined layer\n", getName(), sourceLayerName);
      }
      abort();
   }

   postpostOuterProduct = (pvdata_t *) calloc(nxp*nyp*nfp*getNumWeightPatches(), sizeof(pvdata_t));


   return status;
}

} /* namespace PV */
