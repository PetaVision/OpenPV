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
#include "../src/columns/HyPerCol.hpp"
#include "../src/layers/HyPerLayer.hpp"
#include "../src/connections/HyPerConn.hpp"
// #include "../src/io/ConnectionProbe.hpp"
// #include "../src/io/PostConnProbe.hpp"

#include <assert.h>

using namespace PV;

static int set_weights_to_source_index(HyPerConn * c);
static int check_weights(HyPerConn * c, PVPatch ** postWeights, pvdata_t * dataStart);

int main(int argc, char * argv[])
{
   PVPatch ** postWeights;

   int status = 0;

   HyPerCol  * hc = new HyPerCol("column", argc, argv);
   Example   * l1 = new Example("test_post_weights L1", hc);
   Example   * l2 = new Example("test_post_weights L2", hc);
   Example   * l3 = new Example("test_post_weights L3", hc);
   HyPerConn * c1 = new HyPerConn("test_post_weights L1 to L1", hc, l1, l1, CHANNEL_EXC);
   HyPerConn * c2 = new HyPerConn("test_post_weights L2 to L3", hc, l2, l3, CHANNEL_EXC);
   HyPerConn * c3 = new HyPerConn("test_post_weights L3 to L2", hc, l3, l2, CHANNEL_EXC);
   assert(c1->numberOfAxonalArborLists() == 1);
   assert(c2->numberOfAxonalArborLists() == 1);
   assert(c3->numberOfAxonalArborLists() == 1);

   // set weights to be k index source in pre-synaptic layer
   //
   status = set_weights_to_source_index(c1);
   status = set_weights_to_source_index(c2);
   status = set_weights_to_source_index(c3);

   postWeights = c1->convertPreSynapticWeights(0.0f)[0];
   status = check_weights(c1, postWeights, c1->getWPostData(0,0));
   if (status) return status;

   postWeights = c2->convertPreSynapticWeights(0.0f)[0];
   status = check_weights(c2, postWeights, c2->getWPostData(0,0));
   if (status) return status;

   postWeights = c3->convertPreSynapticWeights(0.0f)[0];
   status = check_weights(c3, postWeights, c3->getWPostData(0,0));
   if (status) return status;

#ifdef DEBUG_PRING
   ConnectionProbe * cp = new ConnectionProbe(-1, -2, 0);
   c2->insertProbe(cp);

   PostConnProbe * pcp = new PostConnProbe(0);
   pcp->setOutputIndices(true);
   c2->insertProbe(pcp);

   c2->outputState(0);
#endif

   return status;
}

static int check_weights(HyPerConn * c, PVPatch ** postWeights, pvdata_t * postDataStart)
{
   int status = 0;

   const int nxPre = c->preSynapticLayer()->clayer->loc.nx;
   const int nyPre = c->preSynapticLayer()->clayer->loc.ny;
   const int nfPre = c->preSynapticLayer()->clayer->loc.nf;
   const int nbPre = c->preSynapticLayer()->clayer->loc.nb;

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
      const int sy = (nxPre + 2*nbPre) * nfPre;
      const int sf = c->fPatchStride(); // p->sf;

      pvdata_t * w = &postDataStart[kPost*c->xPostSize()*c->yPostSize()*c->fPostSize() + p->offset]; // p->data;

      c->preSynapticPatchHead(kxPost, kyPost, kfPost, &kxPre, &kyPre);

      // convert to extended indices
      //
      kxPre += nbPre;
      kyPre += nbPre;

      int kPreHead = kIndex(kxPre, kyPre, kfPre, nxPre+2*nbPre, nyPre+2*nbPre, nfPre);

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

               int ws = (int) w[kp];
               int kPostObserved = ws % numPostPatches;
               int kPreObserved = (ws-kPostObserved)/numPostPatches;

               if (kPre != kPreObserved || kPost != kPostObserved) {
                  status = -1;
                  fprintf(stderr, "ERROR: check_weights: connection %s, kPost==%d kPre==%d kp==%d != w==%d\n",
                          c->getName(), kPost, kPre, kp, (int) w[kp]);
                  fprintf(stderr, "    nxp==%d nyp==%d nfp==%d\n", nxp, nyp, nfp);
                  const char * filename = "post_weights.txt";
                  c->writeTextWeights(filename, kPre);
                  kp++;
                  return status;
               }
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
   short wPacked[2];

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

      pvdata_t * w = c->get_wData(arbor, kPre); // p->data;

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
   if(status==PV_SUCCESS) status = c->writeWeights(filename);

   return status;
}
