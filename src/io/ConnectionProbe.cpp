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
}

ConnectionProbe::ConnectionProbe(int kxPre, int kyPre, int kfPre)
{
   this->kxPre = kxPre;
   this->kyPre = kyPre;
   this->kfPre = kfPre;
   this->kPre  = -1;
   this->fp    = stdout;
}

ConnectionProbe::ConnectionProbe(const char * filename, int kPre)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s%s", OUTPUT_PATH, filename);

   this->kPre = kPre;
   this->fp   = fopen(path, "w");
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

   const PVLayer * l = c->preSynapticLayer()->clayer;

   float nx = l->loc.nx;
   float ny = l->loc.ny;
   float nf   = l->numFeatures;

   // convert to extended frame
   if (kPre < 0) {
      // calculate kPre
      kPre = kIndex((float) kxPre, (float) kyPre, (float) kfPre, nx, ny, nf);
      kPre = kIndexExtended(kPre, nx, ny, nf, l->loc.nPad);
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

   return 0;
}

int ConnectionProbe::text_write_patch(FILE * fd, PVPatch * patch, float * data)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  //assert(sy == nf*nx); // stride could be weird at border
   const int sf = (int) patch->sf;  assert(sf == 1);

   assert(fd != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fd, "%5.3f ", data[i*sx + j*sy + f*sf]);
         }
      }
   }

   return 0;
}

} // namespace PV
