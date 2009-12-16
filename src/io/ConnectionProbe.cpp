/*
 * ConnectionProbe.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: rasmussn
 */

#include "ConnectionProbe.hpp"
#include <assert.h>

namespace PV {

ConnectionProbe::ConnectionProbe(int kPre)
{
   this->kxPre = 0;
   this->kyPre = 0;
   this->kfPre = 0;
   this->kPre  = kPre;
   this->fp    = stdout;
   this->outputIndices = false;
}

ConnectionProbe::ConnectionProbe(int kxPre, int kyPre, int kfPre)
{
   this->kxPre = kxPre;
   this->kyPre = kyPre;
   this->kfPre = kfPre;
   this->kPre  = -1;
   this->fp    = stdout;
   this->outputIndices = false;
}

ConnectionProbe::ConnectionProbe(const char * filename, int kPre)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s%s", OUTPUT_PATH, filename);

   this->kPre = kPre;
   this->fp   = fopen(path, "w");
   this->outputIndices = false;
}

ConnectionProbe::ConnectionProbe(const char * filename, int kxPre, int kyPre, int kfPre)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s%s", OUTPUT_PATH, filename);
   this->fp   = fopen(path, "w");

   this->kxPre = kxPre;
   this->kyPre = kyPre;
   this->kfPre = kfPre;
   this->kPre  = -1;

   this->outputIndices = false;
}
ConnectionProbe::~ConnectionProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int ConnectionProbe::outputState(float time, HyPerConn * c)
{
   float * M = NULL;
   int kPre = this->kPre;

   const PVLayer * lPre = c->preSynapticLayer()->clayer;

   const float nx = lPre->loc.nx;
   const float ny = lPre->loc.ny;
   const float nf = lPre->numFeatures;

   // convert to extended frame
   if (kPre < 0) {
      // calculate kPre
      kPre = kIndex((float) kxPre, (float) kyPre, (float) kfPre, nx, ny, nf);
      kPre = kIndexExtended(kPre, nx, ny, nf, lPre->loc.nPad);
   }

   const int axonId = 0;
   PVAxonalArbor * arbor = c->axonalArbor(kPre, axonId);

   PVPatch * P = arbor->plasticIncr;
   PVPatch * w = arbor->weights;
   int kPost = arbor->offset;

   if (c->getPlasticityDecrement() != NULL) {
      M = & (c->getPlasticityDecrement()->data[kPost]);  // STDP decrement variable
   }

   fprintf(fp, "w%d:      ", kPre);

   if (P != NULL && M != NULL) {
      fprintf(fp, "M= ");
      text_write_patch(fp, P, M);
   }
   if (P != NULL) {
      fprintf(fp, "P= ");
      text_write_patch(fp, P, P->data);  // write the P variable
   }
   fprintf(fp, "w= ");
   text_write_patch(fp, w, w->data);
   fprintf(fp, "\n");
   fflush(fp);

   if (outputIndices) {
      const PVLayer * lPost = c->postSynapticLayer()->clayer;

      const int xScale = lPost->xScale - lPre->xScale;
      const int yScale = lPost->yScale - lPre->yScale;

      // global non-extended post-synaptic frame but I think is
      // local if kxPost0Left and kyPost0Left are zero.
      //
      float kxPost0Left = 0.0;
      float kyPost0Left = 0.0;

      float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, w->nx);
      float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, w->ny);

      write_patch_indices(fp, w, &lPost->loc, kxPost, kyPost, 0);
      fflush(fp);
   }

   return 0;
}

int ConnectionProbe::text_write_patch(FILE * fp, PVPatch * patch, float * data)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  //assert(sy == nf*nx); // stride could be weird at border
   const int sf = (int) patch->sf;  assert(sf == 1);

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
 * The inputs to the function (patch,loc)can either be from
 * the point of view of the pre- or post-synaptic layer.
 *
 * @patch the patch to iterate over
 * @loc the location information in the layer that the patch projects to
 * @nf the number of features in the patch (should be the same as in the layer)
 * @kx0 the kx index location of the head (neuron) of the patch projection
 * @ky0 the ky index location of the head of the patch projection
 * @kf0 the kf index location of the head of the patch (should be 0)
 *
 * NOTE: indices are in the local space
 */
int ConnectionProbe::write_patch_indices(FILE * fp, PVPatch * patch,
                                         const LayerLoc * loc, int kx0, int ky0, int kf0)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

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
            fprintf(fp, "(%4d, (%4d,%4d,%4d)) ", k, kx, ky, kf);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}

} // namespace PV
