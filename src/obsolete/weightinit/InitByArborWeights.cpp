/*
 * InitByArbor.cpp
 *
 *      Author: slundquist 
 */

#include "InitByArborWeights.hpp"

namespace PV {

InitByArborWeights::InitByArborWeights(HyPerConn * conn) : InitWeights() {

   InitByArborWeights::initialize_base();
   InitByArborWeights::initialize(conn);
}

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

int InitByArborWeights::initialize(HyPerConn * conn) {
   int status = InitWeights::initialize(conn);
   return status;
}

int InitByArborWeights::calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId) {

   const int nxp = weightParams->getnxPatch();
   const int nyp = weightParams->getnyPatch();
   const int nfp = weightParams->getnfPatch();

   const int sxp = weightParams->getsx();
   const int syp = weightParams->getsy();
   const int sfp = weightParams->getsf();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            float val = (float)arborId / (nxp * nyp * nfp);
            dataStart[x * sxp + y * syp + f * sfp] = val;
         }
      }
   }


   return PV_SUCCESS;
}

}
