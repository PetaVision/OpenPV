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
   this->kPre = kPre;
   this->fp   = stdout;
}

ConnectionProbe::ConnectionProbe(const char * filename, int kPre)
{
   char path[PV_PATH_MAX];
   sprintf(path, "%s%s", OUTPUT_PATH, filename);

   this->kPre = kPre;
   this->fp   = fopen(path, "w");
}

ConnectionProbe::~ConnectionProbe()
{
   if (fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int ConnectionProbe::outputState(float time, HyPerConn * c)
{
   PVSynapseBundle * tasks = c->tasks(kPre, 0);
   PVPatch * P   = tasks->tasks[0]->plasticIncr;
   PVPatch * w   = tasks->tasks[0]->weights;
   size_t offset = tasks->tasks[0]->offset;

   assert(c->numberOfBundles() == 1);

   float * M = & (c->getPlasticityDecrement()->data[offset]);  // STDP decrement variable

   fprintf(fp, "w%d:      M=", kPre);
   text_write_patch(fp, P, M);
   fprintf(fp, "P=");
   text_write_patch(fp, P, P->data);  // write the P variable
   fprintf(fp, "w=");
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
   const int sy = (int) patch->sy;  //assert(sy == nf*nx);
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
