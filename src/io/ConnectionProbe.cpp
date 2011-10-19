/*
 * ConnectionProbe.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: Craig Rasmussen
 */

#include "ConnectionProbe.hpp"
#include "../connections/STDPConn.hpp"
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
   this->kxPre = 0;
   this->kyPre = 0;
   this->kfPre = 0;
   this->kPre  = kPre;
   this->fp    = stdout;
   this->outputIndices = false;
   this->stdpVars = true;
   this->arborID=arbID;
}

ConnectionProbe::ConnectionProbe(int kxPre, int kyPre, int kfPre, int arbID)
{
   this->kxPre = kxPre;
   this->kyPre = kyPre;
   this->kfPre = kfPre;
   this->kPre = -1;
   this->fp = stdout;
   this->outputIndices = false;
   this->stdpVars = true;
   this->arborID=arbID;
}

ConnectionProbe::ConnectionProbe(const char * filename, HyPerCol * hc, int kPre, int arbID)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s/%s", hc->getOutputPath(), filename);

   this->kPre = kPre;
   this->fp = fopen(path, "w");
   this->outputIndices = false;

   this->stdpVars = true;
   this->arborID=arbID;
}

ConnectionProbe::ConnectionProbe(const char * filename, HyPerCol * hc, int kxPre, int kyPre, int kfPre, int arbID)
{
   const char * outputPath = hc->getOutputPath();
   size_t outputpathlen = strlen(outputPath);
   size_t filenamelen = strlen(filename);
   size_t pathlen = outputpathlen + filenamelen;
   if( pathlen >= PV_PATH_MAX || pathlen < outputpathlen || pathlen < filenamelen || pathlen + 2 <= pathlen) {
      fprintf(stderr, "ConnectionProbe: path to output file too long.  Exiting.\n");
      exit(EXIT_FAILURE);
   }
   char * path = (char *) malloc((pathlen+2)*sizeof(char));
   sprintf(path, "%s/%s", outputPath, filename);
   this->fp   = fopen(path, "w");
   if( !this->fp ) {
      fprintf(stderr, "ConnectionProbe: Unable to open \"%s\" for writing.  Error %d\n", path, errno);
      exit(EXIT_FAILURE);
   }

   this->kxPre = kxPre;
   this->kyPre = kyPre;
   this->kfPre = kfPre;
   this->kPre  = -1;

   this->outputIndices = false;
   this->stdpVars = true;
   this->arborID=arbID;
}
ConnectionProbe::~ConnectionProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
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
         text_write_patch(fp, P, M);
      }
      if (P != NULL) {
         fprintf(fp, "P= ");
         text_write_patch(fp, P, P->data); // write the P variable
      }
      fprintf(fp, "w= ");
      text_write_patch(fp, w, w->data);
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

int ConnectionProbe::text_write_patch(FILE * fp, PVPatch * patch, float * data)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = patch->nf;

   const int sx = patch->sx;  assert(sx == nf);
   const int sy = patch->sy;  //assert(sy == nf*nx); // stride could be weird at border
   const int sf = patch->sf;  assert(sf == 1);

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
   const int nf = patch->nf;

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
