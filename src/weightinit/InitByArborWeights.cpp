/*
 * InitByArbor.cpp
 *
 *      Author: slundquist 
 */

#include "InitByArborWeights.hpp"

namespace PV {

InitByArborWeights::InitByArborWeights()
{
   initialize_base();
}

InitByArborWeights::~InitByArborWeights()
{
}

int InitByArborWeights::initialize_base() {
   return PV_SUCCESS;
}

//InitWeightsParams * InitUniformRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
//   InitWeightsParams * tempPtr = new InitUniformRandomWeightsParams(callingConn);
//   return tempPtr;
//}

int InitByArborWeights::calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {

   const int nxp = weightParams->getnxPatch_tmp(); // wp->nx;
   const int nyp = weightParams->getnyPatch_tmp(); // wp->ny;
   const int nfp = weightParams->getnfPatch_tmp(); //wp->nf;

   const int sxp = weightParams->getsx_tmp(); //wp->sx;
   const int syp = weightParams->getsy_tmp(); //wp->sy;
   const int sfp = weightParams->getsf_tmp(); //wp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            float val = (float)arborId / (nxp * nyp * nfp);
            dataStart[x * sxp + y * syp + f * sfp] = val;
         }
      }
   }


   return PV_SUCCESS; // return 1;
}

}
