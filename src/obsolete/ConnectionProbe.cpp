/*
 * ConnectionProbe.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: Craig Rasmussen
 */

#include "ConnectionProbe.hpp"
#include "../connections/STDPConn.hpp"
#include <limits.h>
#include <assert.h>

namespace PV {

/*
 * NOTES:
 *     - kxPre, kyPre, are indices in the restricted space.
 *     - kPre is the linear index which will be computed in the
 *     extended space, which includes margins.
 *
 *
 */
ConnectionProbe::ConnectionProbe(int kPre, int arbID)
{
   initialize_base();
   initialize(NULL, NULL, INDEX_METHOD, kPre, INT_MIN, INT_MIN, INT_MIN, arbID);
}

ConnectionProbe::ConnectionProbe(int kxPre, int kyPre, int kfPre, int arbID)
{
   initialize_base();
   initialize(NULL, NULL, COORDINATE_METHOD, INT_MIN, kxPre, kyPre, kfPre, arbID);
}

ConnectionProbe::ConnectionProbe(const char * filename, HyPerCol * hc, int kPre, int arbID)
{
   initialize_base();
   initialize(filename, hc, INDEX_METHOD, kPre, INT_MIN, INT_MIN, INT_MIN, arbID);
}

ConnectionProbe::ConnectionProbe(const char * filename, HyPerCol * hc, int kxPre, int kyPre, int kfPre, int arbID)
{
   initialize_base();
   initialize(filename, hc, COORDINATE_METHOD, INT_MIN, kxPre, kyPre, kfPre, arbID);
}

ConnectionProbe::~ConnectionProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int ConnectionProbe::initialize_base() {
   return PV_SUCCESS;
}

int ConnectionProbe::initialize(const char * filename, HyPerCol * hc, ConnectionProbeIndexMethod method, int kPre, int kxPre, int kyPre, int kfPre, int arbID) {
   probeIndexMethod = method;
   if( method == INDEX_METHOD ) {
      this->kPre = kPre;
   }
   else if( method == COORDINATE_METHOD ) {
      this->kxPre = kxPre;
      this->kyPre = kyPre;
      this->kfPre = kfPre;
   }
   this->outputIndices = false;
   this->stdpVars = true;
   this->arborID=arbID;
   return BaseConnectionProbe::initialize(NULL, filename, hc);
}

/**
 * kPre lives in the extended space
 *
 * NOTES:
 *    - kPre is the linear index of the neuron in the extended space.
 *
 */
int ConnectionProbe::outputState(float time, HyPerConn * c)
{
   pvdata_t * M = NULL;
   int kPre = this->kPre;

   const PVLayerLoc * lPreLoc = c->preSynapticLayer()->getLayerLoc();

   const int nxPre = lPreLoc->nx;
   const int nyPre = lPreLoc->ny;
   const int nfPre = lPreLoc->nf;

   // convert to extended frame
   if (kPre < 0) {
      // calculate kPre
      kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);
      kPre = kIndexExtended(kPre, nxPre, nyPre, nfPre, lPreLoc->nb);
   }

   fprintf(fp, "w%d:      \n", kPre);

   //const int axonId = 0;

   //probe only outputs one arbor.  to read more arbors add more probes!
   // PVAxonalArbor * arbor = c->axonalArbor(kPre, arborID);

   PVPatch * P = c->getPlasticIncr(kPre,arborID);
   PVPatch * w = c->getWeights(kPre, arborID);
   int kPost = c->getAPostOffset(kPre, arborID);

   if (stdpVars) {
      STDPConn * stdp_conn = dynamic_cast<STDPConn *>(c);

      if (stdp_conn->getPlasticityDecrement() != NULL) {
         M = &(stdp_conn->getPlasticityDecrement()->data[kPost]); // STDP decrement variable
      }

      if (P != NULL && M != NULL) {
         fprintf(fp, "M= ");
         text_write_patch(fp, P, M, c);
      }
      if (P != NULL) {
         fprintf(fp, "P= ");
         text_write_patch(fp, P, P->data, c); // write the P variable
      }
      fprintf(fp, "w= ");
      text_write_patch(fp, w, w->data, c);
      fprintf(fp, "\n");
      fflush(fp);
   } // if (stdpVars)

   if (outputIndices) {
      const PVLayerLoc * lPostLoc = c->postSynapticLayer()->getLayerLoc();

      const int nxPostExt = lPostLoc->nx + 2*lPostLoc->nb;
      const int nyPostExt = lPostLoc->ny + 2*lPostLoc->nb;
      const int nfPost = lPostLoc->nf;

      //const int kxPost = kxPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      //const int kyPost = kyPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      int kxPost = kxPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->nb;
      int kyPost = kyPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->nb;

      //
      // The following is incorrect because w->nx is reduced near boundary.
      // Remove when verified.
      //
      //int kxPost = zPatchHead(kxPre, w->nx, lPre->xScale, lPost->xScale);
      //int kyPost = zPatchHead(kyPre, w->ny, lPre->yScale, lPost->yScale);


      write_patch_indices(fp, w, lPostLoc, kxPost, kyPost, 0);
      fflush(fp);
   } // if(outputIndices)

   return 0;
}

int ConnectionProbe::text_write_patch(FILE * fp, PVPatch * patch, float * data, HyPerConn * parentConn)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = parentConn->fPatchSize(); //patch->nf;

   const int sx = parentConn->xPatchStride(); //patch->sx;
   assert(sx == nf);
   const int sy = parentConn->yPatchStride(); //patch->sy;  //
   assert(sy == nf*parentConn->xPatchSize()); // stride could be weird at border
   const int sf = parentConn->fPatchStride(); //patch->sf;
   assert(sf == 1);

   assert(fp != NULL);

   for (f = 0; f < nf; f++) {
      fprintf(fp, "f = %i\n  ", f);
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fp, "%5.3f ", data[i*sx + j*sy + f*sf]);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}

/**
 * Write out the layer indices of the positions in a patch.
 * The inputs to the function (patch,loc) can either be from
 * the point of view of the pre- or post-synaptic layer.
 *
 * @patch the patch to iterate over
 * @loc the location information in the layer that the patch projects to
 * @nf the number of features in the patch (should be the same as in the layer)
 * @kx0 the kx index location of the head (neuron) of the patch projection
 * @ky0 the ky index location of the head of the patch projection
 * @kf0 the kf index location of the head of the patch (should be 0)
 *
 * NOTES:
 *    - indices are in the local, restricted space.
 *    - kx0, ky0, are pre patch heads.
 *
 */
int ConnectionProbe::write_patch_indices(FILE * fp, PVPatch * patch,
      const PVLayerLoc * loc, int kx0, int ky0, int kf0)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = loc->nf; //patch->nf;

   // these strides are from the layer, not the patch
   // NOTE: assumes nf from layer == nf from patch
   //
   const int sx = nf;
   const int sy = loc->nx * nf;

   assert(fp != NULL);

   const int k0 = kIndex(kx0, ky0, kf0, loc->nx, loc->ny, nf);

   fprintf(fp, "  ");

   // loop over patch indices (writing out layer indices)
   //
   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            int kf = f;
            int kx = kx0 + i;
            int ky = ky0 + j;
            int k  = k0 + kf + i*sx + j*sy;
            //fprintf(fp, "(%4d, (%4d,%4d,%4d)) ", k, kx, ky, kf);
            fprintf(fp, "%4d %4d %4d %4d  ", k, kx, ky, kf);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}

} // namespace PV
