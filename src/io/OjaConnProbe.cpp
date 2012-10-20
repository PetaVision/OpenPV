/*
 * OjaConnProbe.cpp
 *
 *  Created on: Oct 15, 2012
 *      Author: dpaiton
 */

#include "OjaConnProbe.hpp"

namespace PV {

OjaConnProbe::OjaConnProbe() {
   initialize_base();
}

OjaConnProbe::OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int postIndex, bool isPostProbe)
{
   initialize_base();
   initialize(probename, filename, conn, INDEX_METHOD, postIndex, -1, -1, -1, isPostProbe);
}

OjaConnProbe::OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPost, int kyPost, int kfPost, bool isPostProbe)
{
   initialize_base();
   initialize(probename, filename, conn, COORDINATE_METHOD, -1, kxPost, kyPost, kfPost, isPostProbe);
}

OjaConnProbe::~OjaConnProbe()
{
   free(preStdpTrs);
   free(preOjaTrs);
   free(preWeights);
}

int OjaConnProbe::initialize_base() {
   preStdpTrs = NULL;
   preOjaTrs  = NULL;
   preWeights = NULL;
   kLocal      = -1;
   return PV_SUCCESS;
}

int OjaConnProbe::initialize(const char * probename, const char * filename,
      HyPerConn * conn, PatchIDMethod pidMethod, int kPost,
      int kxPost, int kyPost, int kfPost, bool isPostProbe)
{
   ojaConn = dynamic_cast<OjaSTDPConn *>(conn);
   assert(ojaConn != NULL);

   const PVLayerLoc * postLoc;
   postLoc = ojaConn->postSynapticLayer()->getLayerLoc();

   int nxGlobal = postLoc->nxGlobal;
   int nyGlobal = postLoc->nyGlobal;
   int nf = postLoc->nf;

   if (pidMethod == INDEX_METHOD) {
      kxPost = kxPos(kPost,nxGlobal,nyGlobal,nf);
      kyPost = kyPos(kPost,nxGlobal,nyGlobal,nf);
      kfPost = featureIndex(kPost,nxGlobal,nyGlobal,nf);
   }
   else if(pidMethod == COORDINATE_METHOD) {
      // assert(kfPost != 0); //TODO: Why does a single feature not have a kf of 0? // it doesn't?
      kPost = kIndex(kxPost,kyPost,kfPost,nxGlobal,nyGlobal,nf); // nx, ny, nf NOT in extended space
   }
   else assert(false);
   assert(kPost != -1);
   assert(kfPost <= nf);

   BaseConnectionProbe::initialize(probename, filename, conn,kxPost,kyPost,kfPost,isPostProbe);

// Now convert from global coordinates to local coordinates
   int kxPostLocal = kxPost - postLoc->kx0;
   int kyPostLocal = kyPost - postLoc->ky0;
   int nxLocal = postLoc->nx;
   int nyLocal = postLoc->ny;
   kLocal = kIndex(kxPostLocal,kyPostLocal,kfPost,nxLocal,nyLocal,nf);
   // assert(kLocal >=0 && kLocal < ojaConn->postSynapticLayer()->getNumExtended());

   inBounds = !(kxPostLocal < 0 || kxPostLocal >= postLoc->nx || kyPostLocal < 0 || kyPostLocal >= postLoc->ny);


   return PV_SUCCESS;
}

int OjaConnProbe::outputState(float timef)
{
   if (!inBounds) {
      return PV_SUCCESS;
   }

   // Get post layer sizes
   int nxpPost = ojaConn->getNxpPost();
   int nypPost = ojaConn->getNypPost();
   int nfpPost = ojaConn->getNfpPost();

   int numArbors = ojaConn->numberOfAxonalArborLists(); //will loop through arbors
   int numPostPOVPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   InterColComm * icComm = ojaConn->getParent()->icCommunicator();
   const int rank = icComm->commRank();

   // Allocate buffers for pre info
   preStdpTrs = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   preOjaTrs  = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   preWeights = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   assert(preStdpTrs != NULL);
   assert(preOjaTrs != NULL);
   assert(preWeights != NULL);

   int num_weights_in_patch = ojaConn->xPatchSize()*ojaConn->yPatchSize()*ojaConn->fPatchSize();
   int preTraceIdx = 0;
   for (int arborID=0; arborID < numArbors; arborID++)
   {
      postWeights = ojaConn->getPostWeights(arborID,kLocal); // Pointer array full of addresses pointing to the weights for all of the preNeurons connected to the given postNeuron's receptive field
      float * startAdd = ojaConn->get_wDataStart(arborID);                    // Address of first preNeuron in pre layer
      for (int preNeuronID=0; preNeuronID<numPostPOVPatch; preNeuronID++)
      {
         float * kPreAdd = postWeights[preNeuronID];  // Address of first preNeuron in receptive field of postNeuron
         assert(kPreAdd != NULL);
         int kPre = (kPreAdd-startAdd) / num_weights_in_patch;

         assert(preTraceIdx < numArbors*numPostPOVPatch);
         preWeights[preTraceIdx] = *(postWeights[preNeuronID]); // One weight per arbor per preNeuron in postNeuron's receptive field
         preStdpTrs[preTraceIdx] = ojaConn->getPreStdpTr(kPre); // Trace with STDP-related time scale (tauLTD)
         preOjaTrs[preTraceIdx]  = ojaConn->getPreOjaTr(kPre);  // Trace with Oja-related time scale (tauOja)
         preTraceIdx++;
      }
   }

   postStdpTr  = ojaConn->getPostStdpTr(kLocal);
   postOjaTr   = ojaConn->getPostOjaTr(kLocal);
   postIntTr   = ojaConn->getPostIntTr(kLocal);
   ampLTD      = ojaConn->getAmpLTD(kLocal);

   // Write out to file
   FILE * fp = getFilePtr();
   assert(fp); // invalid pointer

   const char * msg = getName(); // Message to precede the probe's output line

   fprintf(fp, "%s:      t=%.1f kLocal=%d", msg, timef, kLocal);
   fprintf(fp, " poStdpTr=%-6.3f",postStdpTr);
   fprintf(fp, " poOjaTr=%-6.3f",postOjaTr);
   fprintf(fp, " poIntTr=%-6.3f",postIntTr);
   fprintf(fp, " ampLTD=%-6.3f",ampLTD);
   for (int weightIdx=0; weightIdx < numArbors*numPostPOVPatch; weightIdx++) {
      fprintf(fp, " prStdpTr%d=%-6.3f",weightIdx,preStdpTrs[weightIdx]);
      fprintf(fp, " prOjaTr%d=%-6.3f",weightIdx,preOjaTrs[weightIdx]);
      fprintf(fp, " weight%d=%-6.3f",weightIdx,preWeights[weightIdx]);
   }
   fprintf(fp, "\n");
   fflush(fp);

   return PV_SUCCESS;
}
} //end of namespacePV




