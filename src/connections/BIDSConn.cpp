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

BIDSConn::BIDSConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   initialize_base();
   HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
}

int BIDSConn::initialize_base() {
   lateralRadius = 0.0;
   jitterSourceName = NULL;
   jitter = 0.0;
   return PV_SUCCESS;
}

int BIDSConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_lateralRadius(ioFlag);
   ioParam_jitterSource(ioFlag);
   ioParam_jitter(ioFlag);
   return status;
}

void BIDSConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // nxp depends on presynaptic density, so it is handled by setPatchSize in the communicateInitInfo stage
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   return;
}

void BIDSConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // nyp depends on presynaptic density, so it is handled by setPatchSize in the communicateInitInfo stage
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   return;
}

//@lateralRadius: the radius of the mathematical patch in 64x64 space
void BIDSConn::ioParam_lateralRadius(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "lateralRadius", &lateralRadius);
}

void BIDSConn::ioParam_jitterSource(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "jitterSource", &jitterSourceName);
}

//@jitter: The maximum possible amount that a physical node in 256x256 can be placed from its original mathematical position in 256x256 space
//In order to get the full length of the radius at which a node can see its neighboring nodes in 256x256 physical space while accounting for jitter
//on both ends, we take into acct. the provided lateral radius, maximum jitter from the principle node, and maximum jitter from the furthest possible
//neighboring node. Since this occurs on both sides of the patch, the equation is multiplied by two.
void BIDSConn::ioParam_jitter(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "jitter", &jitter);
}

int BIDSConn::setPatchSize()
{
   int xScalePre = pre->getXScale();
   int xScalePost = post->getXScale();
   int xScale = (int)pow(2, xScalePre);
   //Convert to bids space, +1 to round up
   nxp = (1 + 2*(int)(ceil(lateralRadius/(double)xScale) + ceil(2.0 * jitter/(double)xScale)));

   int yScalePre = pre->getYScale();
   int yScalePost = post->getYScale();
   int yScale = (int)pow(2, yScalePre);
   //Convert to bids space, +1 to round up
   nyp = (1 + 2*(int)(ceil(lateralRadius/(double)yScale) + ceil(2.0 * jitter/(double)yScale)));

   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "ny", nyp);
   return PV_SUCCESS;
}

BIDSConn::~BIDSConn() {
   free(jitterSourceName); jitterSourceName = NULL;
}

} // namespace PV
