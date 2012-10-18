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

OjaConnProbe::OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int postIndex)
{
   initialize_base();
   initialize(probename, filename, conn, INDEX_METHOD, postIndex, -1, -1, -1);
}

OjaConnProbe::OjaConnProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPost, int kyPost, int kfPost)
{
   initialize_base();
   initialize(probename, filename, conn, COORDINATE_METHOD, -1, kxPost, kyPost, kfPost);
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
   kPost      = -1;
   return PV_SUCCESS;
}

int OjaConnProbe::initialize(const char * probename, const char * filename,
      HyPerConn * conn, PatchIDMethod pidMethod, int postIndex,
      int kxPost, int kyPost, int kfPost)
{
   BaseConnectionProbe::initialize(probename, filename, conn);

   ojaConn = dynamic_cast<OjaSTDPConn *>(conn);
   assert(ojaConn != NULL);

   postLoc = ojaConn->postSynapticLayer()->getLayerLoc();

   if (pidMethod == INDEX_METHOD) {
      kPost = postIndex;
      //std::cout << "OjaConnProbe: kPost (INDEX): " << kPost << "\n";
   }
   else if(pidMethod == COORDINATE_METHOD) {
      assert(kfPost != 0); //TODO: Why does a single feature not have a kf of 0?
      kPost = kIndex(kxPost,kyPost,kfPost,postLoc->nx,postLoc->ny,postLoc->nf);// nx, ny, nf NOT in extended space
      //std::cout << "OjaConnProbe: kPost (COORDINATE): " << kPost << "\n";
   }
   else assert(false);
   assert(kPost != -1);

   return PV_SUCCESS;
}

int OjaConnProbe::outputState(float timef)
{
   // Get post layer sizes
   int nxpPost = ojaConn->getNxpPost();
   int nypPost = ojaConn->getNypPost();
   int nfpPost = ojaConn->getNfpPost();

   int numArbors = ojaConn->numberOfAxonalArborLists(); //will loop through arbors
   int numPostPOVPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

//   std::cout << "OjaConnProbe: numArbors: " << numArbors << "\n";
//   std::cout << "OjaConnProbe: numPostPOVPatch: " << numPostPOVPatch << "\n";

   // Allocate buffers for pre info
   preStdpTrs = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   preOjaTrs  = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   preWeights = (float *) calloc(numPostPOVPatch*numArbors, sizeof(float));
   assert(preStdpTrs != NULL);
   assert(preOjaTrs != NULL);
   assert(preWeights != NULL);

   const PVLayerLoc * preLoc  = ojaConn->preSynapticLayer()->getLayerLoc(); //need pre layer sizes for patch index info

   int preTraceIdx = 0;
   for (int arborID=0; arborID < numArbors; arborID++)
   {
      postWeights = ojaConn->getPostWeights(arborID,kPost); // Pointer array full of addresses pointing to the weights for all of the preNeurons connected to the given postNeuron's receptive field
      float * startAdd = ojaConn->get_wDataStart(arborID);                    // Address of first preNeuron in pre layer
      for (int preNeuronID=0; preNeuronID<numPostPOVPatch; preNeuronID++)
      {
         float * kPreAdd = postWeights[preNeuronID];                             // Address of first preNeuron in receptive field of postNeuron
         int kPre = (kPreAdd-startAdd) / (preLoc->nx * preLoc->ny * preLoc->nf); // Specific preNeuron's (extended) index

         assert(preTraceIdx < numArbors*numPostPOVPatch);
         preWeights[preTraceIdx] = *(postWeights[preNeuronID]); // One weight per arbor per preNeuron in postNeuron's receptive field
         preStdpTrs[preTraceIdx] = ojaConn->getPreStdpTr(kPre); // Trace with STDP-related time scale (tauLTD)
         preOjaTrs[preTraceIdx]  = ojaConn->getPreOjaTr(kPre);  // Trace with Oja-related time scale (tauOja)
         preTraceIdx++;
      }
   }

   postStdpTr  = ojaConn->getPostStdpTr(kPost);
   postOjaTr   = ojaConn->getPostOjaTr(kPost);
   postIntTr   = ojaConn->getPostIntTr(kPost);
   ampLTD      = ojaConn->getAmpLTD(kPost);

   // Write out to file
   FILE * fp = getFilePtr();
   assert(fp);

   const char * msg = getName(); // Message to precede the probe's output line

   fprintf(fp, "%s:      t=%.1f kPost=%d", msg, timef, kPost);
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




