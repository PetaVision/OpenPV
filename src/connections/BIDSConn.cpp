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

int BIDSConn::setPatchSize(const char * filename)
{
   int status;
   PVParams * inputParams = parent->parameters();

   //nxp = (int) inputParams->value(name, "nxp", post->getCLayer()->loc.nx);
   //nyp = (int) inputParams->value(name, "nyp", post->getCLayer()->loc.ny);

   //@lateralRadius: the radius of the mathematical patch in 64x64 space
   //@jitter: The maximum possible amount that a physical node in 256x256 can be placed from its original mathematical position in 256x256 space
   //In order to get the full length of the radius at which a node can see its neighboring nodes in 256x256 physical space while accounting for jitter
   //on both ends, we take into acct. the provided lateral radius, maximum jitter from the principle node, and maximum jitter from the furthest possible
   //neighboring node. Since this occurs on both sides of the patch, the equation is multiplied by two.
   const char * jitterSourceName = inputParams->stringValue(name, "jitterSource");
   std::cout << jitterSourceName;



   nfp = (int) inputParams->value(name, "nfp", post->getCLayer()->loc.nf);
   if( nfp != post->getCLayer()->loc.nf ) {
      fprintf( stderr, "Params file specifies %d features for connection \"%s\",\n", nfp, name );
      fprintf( stderr, "but %d features for post-synaptic layer %s\n",
               post->getCLayer()->loc.nf, post->getName() );
      exit(PV_FAILURE);
   }
   int xScalePre = pre->getXScale();
   int xScalePost = post->getXScale();

   int xScale = pow(2, xScalePre);
   //Convert to bids space, +1 to round up
   nxp = 1 + 2*(ceil((float)inputParams->value(name, "lateralRadius")/(float)xScale) + ceil((float)2 * (float)inputParams->value(jitterSourceName, "jitter")/(float)xScale));

   status = checkPatchSize(nxp, xScalePre, xScalePost, 'x');
   if( status != PV_SUCCESS) return status;

   int yScalePre = pre->getYScale();
   int yScalePost = post->getYScale();

   int yScale = pow(2, yScalePre);
   //Convert to bids space, +1 to round up
   nyp = 1 + 2 * (ceil((float)inputParams->value(name, "lateralRadius")/(float)yScale) + ceil((float)2 * (float)inputParams->value(jitterSourceName, "jitter")/(float)yScale));

   status = checkPatchSize(nyp, yScalePre, yScalePost, 'y');
   if( status != PV_SUCCESS) return status;

//   std::cout <<"BIDSConn: setPatchSize nxy, nyp: " << nxp << ", " << nyp << "\n";

   status = PV_SUCCESS;
   if( filename != NULL ) {
      bool useListOfArborFiles = inputParams->value(name, "useListOfArborFiles", false)!=0;
      bool combineWeightFiles = inputParams->value(name, "combineWeightFiles", false)!=0;
      if( !useListOfArborFiles && !combineWeightFiles) status = patchSizeFromFile(filename);
   }

   return status;
}

} // namespace PV

#ifdef DEBUG_OPENCL

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLayer_recv_synaptic_input.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/HyPerLayer_recv_synaptic_input.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
#endif
