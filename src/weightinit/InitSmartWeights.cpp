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

   int InitSmartWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams) {
      //smart weights doesn't have any params to load and is too simple to
      //actually need to save anything to work on...

      smartWeights(dataStart, patchIndex);
      return PV_SUCCESS; // return 1;
   }

   InitWeightsParams * InitSmartWeights::createNewWeightParams(HyPerConn * callingConn) {
      InitWeightsParams * tempPtr = new InitWeightsParams(callingConn);
      return tempPtr;
   }

   int InitSmartWeights::smartWeights(/* PVPatch * wp */ pvdata_t * dataStart, int k) {
      // pvdata_t * w = wp->data;

      const int nxp = parentConn->xPatchSize(); // wp->nx;
      const int nyp = parentConn->yPatchSize(); // wp->ny;
      const int nfp = parentConn->fPatchSize(); // wp->nf;

      const int sxp = parentConn->xPatchStride(); //wp->sx;
      const int syp = parentConn->yPatchStride(); //wp->sy;
      const int sfp = parentConn->fPatchStride(); //wp->sf;

      // loop over all post-synaptic cells in patch
      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               dataStart[x * sxp + y * syp + f * sfp] = k;
            }
         }
      }

      return 0;
   }

} /* namespace PV */



