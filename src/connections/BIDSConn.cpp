/*
 * BIDSConn.cpp
 *
 *  Created on: Aug 17, 2012
 *      Author: Brennan Nowers
 */

#include "BIDSConn.hpp"

namespace PV {

//Comments in this conn are assuming a HyPerCol size of 256x256 and a bids_node layer of 1/4 the density.
//Adjust numbers accordingly for a given simulation

BIDSConn::BIDSConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
      const char * post_layer_name, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

int BIDSConn::initialize_base() {
   lateralRadius = 0.0;
   jitterSourceName = NULL;
   jitter = 0.0;
   return PV_SUCCESS;
}

int BIDSConn::setParams(PVParams * params) {
   int status = HyPerConn::setParams(params);
   readLateralRadius(params);
   readJitterSource(params);
   readJitter(params);
   return status;
}

//@lateralRadius: the radius of the mathematical patch in 64x64 space
void BIDSConn::readLateralRadius(PVParams * inputParams) {
   lateralRadius = inputParams->value(name, "lateralRadius");
}


 void BIDSConn::readJitterSource(PVParams * inputParams) {
    const char * jitter_source = inputParams->stringValue(name, "jitterSource");
    jitterSourceName = strdup(jitter_source);
 }

//@jitter: The maximum possible amount that a physical node in 256x256 can be placed from its original mathematical position in 256x256 space
//In order to get the full length of the radius at which a node can see its neighboring nodes in 256x256 physical space while accounting for jitter
//on both ends, we take into acct. the provided lateral radius, maximum jitter from the principle node, and maximum jitter from the furthest possible
//neighboring node. Since this occurs on both sides of the patch, the equation is multiplied by two.
void BIDSConn::readJitter(PVParams * inputParams) {
   jitter = inputParams->value(jitterSourceName, "jitter");
}

int BIDSConn::setPatchSize()
{
   int status = PV_SUCCESS;
   PVParams * inputParams = parent->parameters();

   bool patchSizeSet = false;
   if( filename != NULL ) {
      assert(!inputParams->presentAndNotBeenRead(name, "useListOfArborFiles"));
      assert(!inputParams->presentAndNotBeenRead(name, "combineWeightFiles"));
      if( !useListOfArborFiles && !combineWeightFiles) {
         status = patchSizeFromFile(filename);
         if (status == PV_SUCCESS) patchSizeSet = true;
      }
   }

   if (!patchSizeSet) {
      int xScalePre = pre->getXScale();
      int xScalePost = post->getXScale();
      int xScale = (int)pow(2, xScalePre);
      //Convert to bids space, +1 to round up
      nxp = (1 + 2*(int)(ceil(lateralRadius/(double)xScale) + ceil(2.0 * jitter/(double)xScale)));
      status = checkPatchSize(nxp, xScalePre, xScalePost, 'x');

      int yScalePre = pre->getYScale();
      int yScalePost = post->getYScale();
      int yScale = (int)pow(2, yScalePre);
      //Convert to bids space, +1 to round up
      nyp = (1 + 2*(int)(ceil(lateralRadius/(double)yScale) + ceil(2.0 * jitter/(double)yScale)));
      status = checkPatchSize(nyp, yScalePre, yScalePost, 'y');
   }

   return status;
}

BIDSConn::~BIDSConn() {
   free(jitterSourceName); jitterSourceName = NULL;
}

} // namespace PV
