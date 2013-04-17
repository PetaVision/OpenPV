/*
 * BIDSConn.cpp
 *
 *  Created on: Aug 17, 2012
 *      Author: Brennan Nowers
 */

#include "BIDSConn.hpp"

namespace PV {

BIDSConn::BIDSConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

int BIDSConn::readPatchSize(PVParams * params) {
   int status;

   //@lateralRadius: the radius of the mathematical patch in 64x64 space
   double lateralRadius = params->value(name, "lateralRadius");

   //@jitter: The maximum possible amount that a physical node in 256x256 can be placed from its original mathematical position in 256x256 space
   //In order to get the full length of the radius at which a node can see its neighboring nodes in 256x256 physical space while accounting for jitter
   //on both ends, we take into acct. the provided lateral radius, maximum jitter from the principle node, and maximum jitter from the furthest possible
   //neighboring node. Since this occurs on both sides of the patch, the equation is multiplied by two.
   const char * jitterSourceName = params->stringValue(name, "jitterSource");
   double jitter = params->value(jitterSourceName, "jitter");

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

   nxpShrunken = 1;
   nypShrunken = 1;

   return status;
}

int BIDSConn::setPatchSize(const char * filename)
{
   int status = PV_SUCCESS;
   PVParams * inputParams = parent->parameters();

   if( filename != NULL ) {
      bool useListOfArborFiles = inputParams->value(name, "useListOfArborFiles", false)!=0;
      bool combineWeightFiles = inputParams->value(name, "combineWeightFiles", false)!=0;
      if( !useListOfArborFiles && !combineWeightFiles) status = patchSizeFromFile(filename);
   }

   return status;
}

} // namespace PV
