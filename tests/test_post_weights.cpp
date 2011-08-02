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
#include "../src/io/ConnectionProbe.hpp"
#include "../src/io/PostConnProbe.hpp"

#include <assert.h>

using namespace PV;

static int set_weights_to_source_index(HyPerConn * c);
static int check_weights(HyPerConn * c, PVPatch ** postWeights);

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

   // set weights to be k index source in pre-synaptic layer
   //
   status = set_weights_to_source_index(c1);
   status = set_weights_to_source_index(c2);
   status = set_weights_to_source_index(c3);

   postWeights = c1->convertPreSynapticWeights(0.0f);
   status = check_weights(c1, postWeights);
   if (status) return status;

   postWeights = c2->convertPreSynapticWeights(0.0f);
   status = check_weights(c2, postWeights);
   if (status) return status;

   postWeights = c3->convertPreSynapticWeights(0.0f);
   status = check_weights(c3, postWeights);
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

static int check_weights(HyPerConn * c, PVPatch ** postWeights)
{
   int status = 0;

   const int nxPre = c->preSynapticLayer()->clayer->loc.nx;
   const int nyPre = c->preSynapticLayer()->clayer->loc.ny;
   const int nfPre = c->preSynapticLayer()->clayer->loc.nf;
   const int nbPre = c->preSynapticLayer()->clayer->loc.nb;

   const int nx = c->postSynapticLayer()->clayer->loc.nx;
   const int ny = c->postSynapticLayer()->clayer->loc.ny;
   const int nf = c->postSynapticLayer()->clayer->loc.nf;

   const int numPatches = nx * ny * nf;

   // assume (or at least use) only one arbor (set of weight patches)
   for (int kPost = 0; kPost < numPatches; kPost++) {
      int kxPre, kyPre, kfPre = 0;

      const int kxPost = kxPos(kPost, nx, ny, nf);
      const int kyPost = kyPos(kPost, nx, ny, nf);
      const int kfPost = featureIndex(kPost, nx, ny, nf);

      PVPatch * p = postWeights[kPost];

      const int nxp = p->nx;
      const int nyp = p->ny;
      const int nfp = p->nf;

      // these strides are from the extended pre-synaptic layer, not the patch
      // NOTE: assumes nf from layer == nf from patch
      //
      const int sx = nfPre;
      const int sy = (nxPre + 2*nbPre) * nfPre;
      const int sf = p->sf;

      pvdata_t * w = p->data;

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
               short * ws;
               int kPre = kPreHead + i*sx + j*sy + f*sf;

               ws = (short *) &w[kp++];

               if (kPre != (int) ws[0] || kPost != (int) ws[1]) {
                  status = -1;
                  fprintf(stderr, "ERROR: check_weights: kPost==%d kPre==%d kp==%d != w==%d\n",
                          kPost, kPre, kp, (int) w[kp-1]);
                  fprintf(stderr, "    nxp==%d nyp==%d nfp==%d\n", nxp, nyp, nfp);
                  const char * filename = "post_weights.txt";
                  c->writeTextWeights(filename, kPre);
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

   int numPatches = c->numWeightPatches(arbor);

   // assume (or at least use) only one arbor (set of weight patches)
   // k index is in extended space
   for (int kPre = 0; kPre < numPatches; kPre++) {
      int kxPostHead, kyPostHead, kfPostHead;
      int nxp, nyp;
      int dx, dy;
      PVPatch * p = c->getWeights(kPre, arbor);

      c->postSynapticPatchHead(kPre, &kxPostHead, &kyPostHead, &kfPostHead, &dx, &dy, &nxp, &nyp);

      const int nfp = p->nf;

      assert(nxp == p->nx);
      assert(nyp == p->ny);
      assert(nfp == lPost->loc.nf);

      const int sxp = p->sx;
      const int syp = p->sy;
      const int sfp = p->sf;

      pvdata_t * w = p->data;

      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               int kxPost = kxPostHead + x;
               int kyPost = kyPostHead + y;
               int kfPost = kfPostHead + f;
               int kPost = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);

               wPacked[0] = kPre;
               wPacked[1] = kPost;
               w[x*sxp + y*syp + f*sfp] = * ((float *) wPacked);
               //w[x*sxp + y*syp + f*sfp] = kPre;
            }
         }
      }

   } // end loop over weight patches

   return status;
}
