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

LCALIFLateralProbe::LCALIFLateralProbe(const char * probename, HyPerCol * hc)
{
   initialize_base();
   initialize(probename, hc);
}

LCALIFLateralProbe::~LCALIFLateralProbe()
{
   if (inBounds){
      free(preWeights);
   }
}

int LCALIFLateralProbe::initialize_base() {
   postIntTr = -1;
   preWeights = NULL;
   kLocalRes  = -1;
   return PV_SUCCESS;
}

int LCALIFLateralProbe::initialize(const char * probename, HyPerCol * hc) {
   BaseConnectionProbe::initialize(probename, hc);
   PVParams * params = hc->parameters();
   //Since it's a lateral conn, postConn shouldn't matter
   int indexmethod = kPost >= 0;
   int coordmethod = kxPost >= 0 && kyPost >= 0 && kfPost >= 0;;
   if( indexmethod && coordmethod ) {
      fprintf(stderr, "%s \"%s\": Ambiguous definition with both kPost and (kxPost,kyPost,kfPost) defined\n", parent->parameters()->groupKeywordFromName(name), name);
      exit(EXIT_FAILURE);
   }
   if( !indexmethod && !coordmethod ) {
      fprintf(stderr, "%s \"%s\": Exactly one of kPost and (kxPost,kyPost,kfPost) must be defined\n", parent->parameters()->groupKeywordFromName(name), name);
      exit(EXIT_FAILURE);
   }
   if( indexmethod ) {
      patchIDMethod = INDEX_METHOD;
   }
   else {
      assert(coordmethod);
      patchIDMethod = COORDINATE_METHOD;
   }

   return PV_SUCCESS;
}

int LCALIFLateralProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseConnectionProbe::ioParamsFillGroup(ioFlag);
   ioParam_kPost(ioFlag);
   ioParam_kxPost(ioFlag);
   ioParam_kyPost(ioFlag);
   ioParam_kfPost(ioFlag);
   return status;
}

// Since params can specify either kPost or (kxPost,kyPost,kfPost), use -1 as a flag that the value was not supplied
void LCALIFLateralProbe::ioParam_kPost(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "kPost", &kPost, -1, false/*warnIfAbsent*/);
}

void LCALIFLateralProbe::ioParam_kxPost(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "kxPost", &kxPost, -1, false/*warnIfAbsent*/);
}

void LCALIFLateralProbe::ioParam_kyPost(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "kyPost", &kyPost, -1, false/*warnIfAbsent*/);
}

void LCALIFLateralProbe::ioParam_kfPost(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "kfPost", &kfPost, -1, false/*warnIfAbsent*/);
}

int LCALIFLateralProbe::allocateDataStructures() {
   BaseConnectionProbe::allocateDataStructures();
   LCALIFConn = dynamic_cast<LCALIFLateralConn *>(targetConn);
   assert(LCALIFConn != NULL);

   const PVLayerLoc * loc;
   //Connecting to itself, so pre/post is same layer
   loc = LCALIFConn->preSynapticLayer()->getLayerLoc();

   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;

   if (patchIDMethod == INDEX_METHOD) {
      kxPost = kxPos(kPost,nxGlobal,nyGlobal,nf);
      kyPost = kyPos(kPost,nxGlobal,nyGlobal,nf);
      kfPost = featureIndex(kPost,nxGlobal,nyGlobal,nf);
   }
   else if(patchIDMethod == COORDINATE_METHOD) {
      //kfPost can't be lower than 0
      assert(kfPost >= 0);
      kPost = kIndex(kxPost,kyPost,kfPost,nxGlobal,nyGlobal,nf); // nx, ny, nf NOT in extended space
   }
   else assert(false);
   assert(kPost != -1);
   assert(kfPost < nf);

   // Now convert from global coordinates to local coordinates
   //Restricted index
   int kxPostLocal = kxPost - loc->kx0;
   int kyPostLocal = kyPost - loc->ky0;
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;

   //Restricted Index
   kLocalRes = kIndex(kxPostLocal,kyPostLocal,kfPost,nxLocal,nyLocal,nf);
   kLocalExt = kIndexExtended(kLocalRes, nxLocal, nyLocal, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);

   // assert(kLocal >=0 && kLocal < ojaConn->postSynapticLayer()->getNumExtended());

   inBounds = !(kxPostLocal < 0 || kxPostLocal >= loc->nx || kyPostLocal < 0 || kyPostLocal >= loc->ny);

   if (inBounds){
      int numArbors = LCALIFConn->numberOfAxonalArborLists(); //will loop through arbors
      int nxpPost = LCALIFConn->xPatchSize();
      int nypPost = LCALIFConn->yPatchSize();
      int nfpPost = LCALIFConn->fPatchSize();

      int numPrePatch = nxpPost * nypPost * nfpPost;

      // Allocate buffers for pre info
      preWeights = (pvwdata_t *) calloc(numPrePatch*numArbors, sizeof(pvwdata_t));
      assert(preWeights != NULL);
   }
   return PV_SUCCESS;
}

int LCALIFLateralProbe::outputState(double timef)
{
   if (!inBounds) {
      return PV_SUCCESS;
   }

   // Get post layer sizes
   int nxp = LCALIFConn->xPatchSize();
   int nyp = LCALIFConn->yPatchSize();
   int nfp = LCALIFConn->fPatchSize();

   int numArbors = LCALIFConn->numberOfAxonalArborLists(); //will loop through arbors
   int numPostPatch = nxp* nyp* nfp;

   int preTraceIdx = 0;

   for (int arborID=0; arborID < numArbors; arborID++)
   {
      LCALIFConn->convertPreSynapticWeights(timef);

      postWeights = LCALIFConn->getWPostData(arborID, kLocalRes);
      postIntTr = LCALIFConn->getIntegratedSpikeCount(kLocalExt);

      for (int postKPatch=0; postKPatch<numPostPatch; postKPatch++)
      {

         //Check kLocalRes to see if the
         //Weights
         preWeights[preTraceIdx] = postWeights[postKPatch];

//         //Int spike counts
//         int kxPost = kxPos(postKPatch, nxp, nyp, nfp);
//         int kyPost = kyPos(postKPatch, nxp, nyp, nfp);
//         int kPostExt = aPostOffset + kxPost*sxp + kyPost*syp;
//         preIntTrs[preTraceIdx] = LCALIFConn->getIntegratedSpikeCount(kPostExt);

         assert(preTraceIdx < numArbors*numPostPatch);
         preTraceIdx++;
      }
   }

   // Write out to file
   FILE * fp = getStream()->fp;
   assert(fp); // invalid pointer

   const char * msg = getName(); // Message to precede the probe's output line

   fprintf(fp, "%s:      t=%.1f kLocalExt=%d", msg, timef, kLocalExt);
   int weightIdx = 0;
   for (int arborID=0; arborID < numArbors; arborID++) {
      fprintf(fp, " postIntTr=%-6.3f",postIntTr);
      for (int patchID=0; patchID < numPostPatch; patchID++) {
         fprintf(fp, " weight_%d_%d=%-6.3f",arborID,patchID,preWeights[weightIdx]);
         weightIdx++;
      }
   }
   fprintf(fp, "\n");
   fflush(fp);

   return PV_SUCCESS;
}
} //end of namespacePV

