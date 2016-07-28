/**
 * test_post_weights.cpp
 *
 *  Created on: Feb 4, 2010
 *      Author: Craig Rasmussen
 *
 * This file tests the conversion of pre-synaptic weight patches to post-synaptic
 * weight patches.
 *
 */

#undef DEBUG_PRINT

#include "Example.hpp"
#include <columns/HyPerCol.hpp>
#include <layers/HyPerLayer.hpp>
#include <connections/HyPerConn.hpp>
#include <weightinit/InitUniformWeights.hpp>

#include <assert.h>

using namespace PV;

static int set_weights_to_source_index(HyPerConn * c);
static int check_weights(HyPerConn * c, PVPatch ** postWeights, pvdata_t * dataStart);

int main(int argc, char * argv[])
{
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   PVPatch ** postWeights;

   int status = 0;

   const char * l1name = "test_post_weights L1";
   const char * l2name = "test_post_weights L2";
   const char * l3name = "test_post_weights L3";
   HyPerCol  * hc = new HyPerCol("column", initObj);
   Example   * l1 = new Example(l1name, hc); assert(l1);
   Example   * l2 = new Example(l2name, hc); assert(l2);
   Example   * l3 = new Example(l3name, hc); assert(l3);

   HyPerConn * c1 = new HyPerConn("test_post_weights L1 to L1", hc);
   assert(c1);
   assert(c1->numberOfAxonalArborLists() == 1);

   HyPerConn * c2 = new HyPerConn("test_post_weights L2 to L3", hc);
   assert(c2);
   assert(c2->numberOfAxonalArborLists() == 1);

   HyPerConn * c3 = new HyPerConn("test_post_weights L3 to L2", hc);
   assert(c3);
   assert(c3->numberOfAxonalArborLists() == 1);
   
   // We're not calling hc->run() because we don't execute any timesteps.
   // But we still need to allocate the weights, so we call the
   // layers' and connections' communicate and allocate methods externally.

   hc->ensureDirExists(hc->getOutputPath());

   auto objectMap = hc->copyObjectMap();
   auto commMessagePtr = std::make_shared<CommunicateInitInfoMessage<BaseObject*> >(*objectMap);
   for (int l=0; l<hc->numberOfLayers(); l++) {
      HyPerLayer * layer = hc->getLayer(l);
      int status = layer->respond(commMessagePtr);
      assert(status==PV_SUCCESS);
   }
   for (int c=0; c<hc->numberOfConnections(); c++) {
      BaseConnection * conn = hc->getConnection(c);
      int status = conn->respond(commMessagePtr);
      assert(status==PV_SUCCESS);
   }
   delete objectMap;

   auto allocateMessagePtr = std::make_shared<AllocateDataMessage>();
   for (int l=0; l<hc->numberOfLayers(); l++) {
      HyPerLayer * layer = hc->getLayer(l);
      int status = layer->respond(allocateMessagePtr);
      assert(status==PV_SUCCESS);
   }

   for (int c=0; c<hc->numberOfConnections(); c++) {
      BaseConnection * conn = hc->getConnection(c);
      int status = conn->respond(allocateMessagePtr);
      assert(status==PV_SUCCESS);
   }

   // Don't need to call initializeState methods:
   // we don't look at the layer values, and the weight values are
   // set by calling set_weights_to_source_index

   // set weights to be k index source in pre-synaptic layer
   //
   status = set_weights_to_source_index(c1);
   postWeights = c1->convertPreSynapticWeights(0.0f)[0];
   status = c1->writePostSynapticWeights(c1->getParent()->simulationTime(),false);
   status = check_weights(c1, postWeights, c1->getWPostData(0,0));
   if (status) return status;

   status = set_weights_to_source_index(c2);
   postWeights = c2->convertPreSynapticWeights(0.0f)[0];
   status = c2->writePostSynapticWeights(c2->getParent()->simulationTime(),false);
   status = check_weights(c2, postWeights, c2->getWPostData(0,0));
   if (status) return status;

   status = set_weights_to_source_index(c3);
   postWeights = c3->convertPreSynapticWeights(0.0f)[0];
   status = c3->writePostSynapticWeights(c3->getParent()->simulationTime(),false);
   status = check_weights(c3, postWeights, c3->getWPostData(0,0));
   if (status) return status;

#ifdef DEBUG_PRINT
   ConnectionProbe * cp = new ConnectionProbe(-1, -2, 0);
   c2->insertProbe(cp);

   PostConnProbe * pcp = new PostConnProbe(0);
   pcp->setOutputIndices(true);
   c2->insertProbe(pcp);

   c2->outputState(0);
#endif

   delete hc;
   delete initObj;

   return status;
}

static int check_weights(HyPerConn * c, PVPatch ** postWeights, pvdata_t * postDataStart)
{
   int status = 0;

   const int nxPre = c->preSynapticLayer()->clayer->loc.nx;
   const int nyPre = c->preSynapticLayer()->clayer->loc.ny;
   const int nfPre = c->preSynapticLayer()->clayer->loc.nf;
   const PVHalo * halo = &c->preSynapticLayer()->clayer->loc.halo;

   const int nx = c->postSynapticLayer()->clayer->loc.nx;
   const int ny = c->postSynapticLayer()->clayer->loc.ny;
   const int nf = c->postSynapticLayer()->clayer->loc.nf;

   const int numPostPatches = nx * ny * nf;

   // assume (or at least use) only one arbor (set of weight patches)
   for (int kPost = 0; kPost < numPostPatches; kPost++) {
      int kxPre, kyPre, kfPre = 0;

      const int kxPost = kxPos(kPost, nx, ny, nf);
      const int kyPost = kyPos(kPost, nx, ny, nf);
      const int kfPost = featureIndex(kPost, nx, ny, nf);

      PVPatch * p = postWeights[kPost];

      const int nxp = p->nx;
      const int nyp = p->ny;
      const int nfp = c->fPatchSize(); // p->nf;

      // these strides are from the extended pre-synaptic layer, not the patch
      // NOTE: assumes nf from layer == nf from patch
      //
      const int sx = nfPre;
      const int sy = (nxPre + halo->lt + halo->rt) * nfPre;
      const int sf = c->fPatchStride(); // p->sf;

      const int postPatchSize = c->xPostPatchSize()*c->yPostPatchSize()*c->fPostPatchSize();
      pvdata_t * w = &postDataStart[kPost*postPatchSize + p->offset]; // p->data;

      c->preSynapticPatchHead(kxPost, kyPost, kfPost, &kxPre, &kyPre);

      // convert to extended indices
      //
      kxPre += halo->lt;
      kyPre += halo->up;

      int kPreHead = kIndex(kxPre, kyPre, kfPre, nxPre+halo->lt+halo->rt, nyPre+halo->dn+halo->up, nfPre);

      // FIND OUT WHY THIS DOESN'T WORK
      //int kPreHead = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);
      //kPreHead = kIndexExtended(kPreHead, nxPre, nyPre, nfPre, nPadPre);

      // loop over patch indices (writing out layer indices)
      //
      int kp = 0;
      for (int f = 0; f < nfp; f++) {
         for (int j = 0; j < nyp; j++) {
            for (int i = 0; i < nxp; i++) {
               int kPre = kPreHead + i*sx + j*sy + f*sf;

               // short * ws;
               // ws = (short *) &w[kp++];
               // int kPreObserved = (int) ws[0];
               // int kPostObserved = (int) ws[1];

               float ws = w[kp]; // int ws = (int) w[kp];
               int kPostObserved = (int) nearbyintf((float)fmod(ws,numPostPatches));
               int kPreObserved = (int) nearbyintf((ws-kPostObserved)/numPostPatches);

               if (kPre != kPreObserved || kPost != kPostObserved) {
                  status = -1;
                  pvErrorNoExit(errorMessage);
                  errorMessage.printf("check_weights: connection %s, kPost==%d kPre==%d kp==%d, expected %d != w==%d\n",
                          c->getName(), kPost, kPre, kp, kPre*numPostPatches+kPost, (int) w[kp]);
                  errorMessage.printf("    nxp==%d nyp==%d nfp==%d\n", nxp, nyp, nfp);
                  const char * filename = "post_weights.txt";
                  c->writeTextWeights(filename, kPre);
                  return status;
               }
               kp++;
            }
         }
      }

   } // end loop over weight patches

   return status;
}

static int set_weights_to_source_index(HyPerConn * c)
{
   int status = 0;
   int arbor = 0;

   assert(sizeof(short) == 2);

   const PVLayer * lPost = c->postSynapticLayer()->clayer;

   const int nxPost = lPost->loc.nx;
   const int nyPost = lPost->loc.ny;
   const int nfPost = lPost->loc.nf;
   const int numPostPatches = nxPost * nyPost * nfPost;

   int numPatches = c->getNumWeightPatches();

   // assume (or at least use) only one arbor (set of weight patches)
   // k index is in extended space
   for (int kPre = 0; kPre < numPatches; kPre++) {
      int kxPostHead, kyPostHead, kfPostHead;
      int nxp, nyp;
      int dx, dy;
      PVPatch * p = c->getWeights(kPre, arbor);

      c->postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx, &dy, &nxp, &nyp);

      const int nfp = c->fPatchSize(); // p->nf;

      assert(nxp == p->nx);
      assert(nyp == p->ny);
      assert(nfp == lPost->loc.nf);

      const int sxp = c->xPatchStride(); // p->sx;
      const int syp = c->yPatchStride(); // p->sy;
      const int sfp = c->fPatchStride(); // p->sf;

      pvwdata_t * w = c->get_wData(arbor, kPre); // p->data;

      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               int kxPost = kxPostHead + x;
               int kyPost = kyPostHead + y;
               int kfPost = kfPostHead + f;
               int kPost = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
               // wPacked[0] = kPre;
               // wPacked[1] = kPost;
               w[x*sxp + y*syp + f*sfp] = kPre*numPostPatches + kPost; // * ((float *) wPacked);
               //w[x*sxp + y*syp + f*sfp] = kPre;
            }
         }
      }

   } // end loop over weight patches
   char filename[PV_PATH_MAX];
   status = snprintf(filename, PV_PATH_MAX, "%s/%s_W.pvp", c->getParent()->getOutputPath(), c->getName())<PV_PATH_MAX ? PV_SUCCESS : PV_FAILURE;
   if(status == PV_SUCCESS) status = c->writeWeights(filename);
   status = snprintf(filename, PV_PATH_MAX, "%s/%s_W.pvp", c->getParent()->getOutputPath(), c->getName())<PV_PATH_MAX ? PV_SUCCESS : PV_FAILURE;

   return status;
}
