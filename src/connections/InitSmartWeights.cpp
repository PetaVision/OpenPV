/*
 * InitSmartWeights.cpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#include "InitSmartWeights.hpp"

namespace PV {

   InitSmartWeights::InitSmartWeights()
   {
      initialize_base();
   }
//   InitSmartWeights::InitSmartWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
//         ChannelType channel) : InitWeights() {
//
//      InitSmartWeights::initialize_base();
//      InitSmartWeights::initialize(name, hc, pre, post, channel);
//   }

   InitSmartWeights::~InitSmartWeights()
   {
      // TODO Auto-generated destructor stub
   }

   int InitSmartWeights::initialize_base() {
      return PV_SUCCESS;
   }
//   int InitSmartWeights::initialize(const char * name, HyPerCol * hc,
//         HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
//      InitWeights::initialize(name, hc, pre, post, channel);
//      return PV_SUCCESS;
//   }

   int InitSmartWeights::calcWeights(PVPatch * patch, int patchIndex, InitWeightsParams *weightParams) {
      //smart weights doesn't have any params to load and is too simple to
      //actually need to save anything to work on...
      smartWeights(patch, patchIndex);
      return 1;
   }

   InitWeightsParams * InitSmartWeights::createNewWeightParams(HyPerConn * callingConn) {
      InitWeightsParams * tempPtr = new InitWeightsParams(callingConn);
      return tempPtr;
   }

   int InitSmartWeights::smartWeights(PVPatch * wp, int k) {
      pvdata_t * w = wp->data;

      const int nxp = wp->nx;
      const int nyp = wp->ny;
      const int nfp = wp->nf;

      const int sxp = wp->sx;
      const int syp = wp->sy;
      const int sfp = wp->sf;

      // loop over all post-synaptic cells in patch
      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               w[x * sxp + y * syp + f * sfp] = k;
            }
         }
      }

      return 0;
   }

} /* namespace PV */



