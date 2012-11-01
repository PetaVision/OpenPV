/*
 * LCALIFLateralProbe.cpp
 *
 *  Created on: Oct 30, 2012
 *      Author: slundquist
 */

#include "LCALIFLateralProbe.hpp"

namespace PV {

LCALIFLateralProbe::LCALIFLateralProbe() {
   initialize_base();
}

LCALIFLateralProbe::LCALIFLateralProbe(const char * probename, const char * filename, HyPerConn * conn, int preIndex)
{
   initialize_base();
   //Since it's a lateral conn, postConn shouldn't matter
   initialize(probename, filename, conn, INDEX_METHOD, preIndex, -1, -1, -1, false);
}

LCALIFLateralProbe::LCALIFLateralProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPre, int kyPre, int kfPre)
{
   initialize_base();
   initialize(probename, filename, conn, COORDINATE_METHOD, -1, kxPre, kyPre, kfPre, false);
}

LCALIFLateralProbe::~LCALIFLateralProbe()
{
   if (inBounds){
      free(preIntTrs);
      free(preWeights);
   }
}

int LCALIFLateralProbe::initialize_base() {
   preIntTrs = NULL;
   preWeights = NULL;
   kLocalExt      = -1;
   return PV_SUCCESS;
}

int LCALIFLateralProbe::initialize(const char * probename, const char * filename,
      HyPerConn * conn, PatchIDMethod pidMethod, int kPre,
      int kxPre, int kyPre, int kfPre, bool isPostProbe)
{
   LCALIFConn = dynamic_cast<LCALIFLateralConn *>(conn);
   assert(LCALIFConn != NULL);

   const PVLayerLoc * loc;
   //Connecting to itself, so pre/post is same layer
   loc = LCALIFConn->preSynapticLayer()->getLayerLoc();

   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;

   if (pidMethod == INDEX_METHOD) {
      kxPre = kxPos(kPre,nxGlobal,nyGlobal,nf);
      kyPre = kyPos(kPre,nxGlobal,nyGlobal,nf);
      kfPre = featureIndex(kPre,nxGlobal,nyGlobal,nf);
   }
   else if(pidMethod == COORDINATE_METHOD) {
      //kfPost can't be lower than 0
      assert(kfPre >= 0);
      kPre = kIndex(kxPre,kyPre,kfPre,nxGlobal,nyGlobal,nf); // nx, ny, nf NOT in extended space
   }
   else assert(false);
   assert(kPre != -1);
   assert(kfPre < nf);

   BaseConnectionProbe::initialize(probename, filename, conn,kxPre,kyPre,kfPre,isPostProbe);

// Now convert from global coordinates to local coordinates
   //Restricted index
   int kxPreLocal = kxPre - loc->kx0;
   int kyPreLocal = kyPre - loc->ky0;
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;
   int nfLocal = loc->nf;
   int nbLocal = loc->nb;
   //Restricted Index
   int kLocalRes = kIndex(kxPreLocal,kyPreLocal,kfPre,nxLocal,nyLocal,nf);
   kLocalExt = kIndexExtended(kLocalRes, nxLocal, nyLocal, nfLocal, nbLocal);

   // assert(kLocal >=0 && kLocal < ojaConn->postSynapticLayer()->getNumExtended());

   inBounds = !(kxPreLocal < 0 || kxPreLocal >= loc->nx || kyPreLocal < 0 || kyPreLocal >= loc->ny);

   if (inBounds){
      int numArbors = LCALIFConn->numberOfAxonalArborLists(); //will loop through arbors
      //Grab patch information
      //Weights of different arbors should be the same
      PVPatch* prePatch = LCALIFConn->getWeights(kLocalExt,0);
      // Get pre layer sizes
      int nxp = prePatch->nx;
      int nyp = prePatch->ny;
      int nfp = LCALIFConn->fPatchSize();

      //Check for other arbors
      //Should only be one arbor, so it should skip this loop
      //Arbor id starts at 1 since id 0 was set to nxp and nyp
      for (int arborID = 1; arborID < numArbors; arborID++){
         prePatch = LCALIFConn->getWeights(kLocalExt,0);
         assert(nxp == prePatch->nx);
         assert(nyp == prePatch->ny);
      }

      int numPrePatch = nxp * nyp * nfp;

      // Allocate buffers for pre info
      preWeights = (float *) calloc(numPrePatch*numArbors, sizeof(float));
      assert(preWeights != NULL);
      preIntTrs = (float*) calloc(numPrePatch*numArbors, sizeof(float));
      assert(preIntTrs != NULL);
   }
   return PV_SUCCESS;
}

int LCALIFLateralProbe::outputState(double timef)
{
   if (!inBounds) {
      return PV_SUCCESS;
   }

   PVPatch* prePatch = LCALIFConn->getWeights(kLocalExt, 0);

   // Get post layer sizes
   int nxp = prePatch->nx;
   int nyp = prePatch->ny;
   int nfp = LCALIFConn->fPatchSize();
   int sxp = LCALIFConn->getPostExtStrides()->sx;
   int syp = LCALIFConn->getPostExtStrides()->sy;

   int numArbors = LCALIFConn->numberOfAxonalArborLists(); //will loop through arbors
   int numPrePatch = nxp* nyp* nfp;

   int preTraceIdx = 0;

   for (int arborID=0; arborID < numArbors; arborID++)
   {


      float* preWeightsData = LCALIFConn->get_wData(arborID, kLocalExt);
      int aPostOffset = LCALIFConn->getAPostOffset(kLocalExt, arborID);

      for (int postNeuronID=0; postNeuronID<numPrePatch; postNeuronID++)
      {
         //Weights
         preWeights[preTraceIdx] = preWeightsData[postNeuronID];

         //Int spike counts
         int kxPost = kxPos(postNeuronID, nxp, nyp, nfp);
         int kyPost = kyPos(postNeuronID, nxp, nyp, nfp);
         int kPostExt = aPostOffset + kxPost*sxp + kyPost*syp;
         preIntTrs[preTraceIdx] = LCALIFConn->getIntegratedSpikeCount(kPostExt);

         assert(preTraceIdx < numArbors*numPrePatch);
         preTraceIdx++;
      }
   }

   // Write out to file
   FILE * fp = getFilePtr();
   assert(fp); // invalid pointer

   const char * msg = getName(); // Message to precede the probe's output line

   fprintf(fp, "%s:      t=%.1f kLocalExt=%d", msg, timef, kLocalExt);
   int weightIdx = 0;
   for (int arborID=0; arborID < numArbors; arborID++) {
      for (int patchID=0; patchID < numPrePatch; patchID++) {
         fprintf(fp, " weight_%d_%d=%-6.3f",arborID,patchID,preWeights[weightIdx]);
         fprintf(fp, " preIntTrs_%d_%d=%-6.3f",arborID,patchID,preIntTrs[weightIdx]);
         weightIdx++;
      }
   }
   fprintf(fp, "\n");
   fflush(fp);

   return PV_SUCCESS;
}
} //end of namespacePV




