/*
 * ProbeActivityLinear.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LinearActivityProbe.hpp"

namespace PV {

LinearActivityProbe::LinearActivityProbe(HyPerCol * hc, PVDimType dim, int loc, int f)
   : PVLayerProbe()
{
   this->parent = hc;
   this->dim = dim;
   this->loc = loc;
   this->f   = f;
}

LinearActivityProbe::LinearActivityProbe(const char * filename, HyPerCol * hc, PVDimType dim, int loc, int f)
    : PVLayerProbe(filename)
{
   this->parent = hc;
   this->dim = dim;
   this->loc = loc;
   this->f   = f;
}

int LinearActivityProbe::outputState(float time, PVLayer * l)
{
   int nk, sk;
   float * line;

   float dt = parent->getDeltaTime();
   int nf = l->numFeatures;

   if (dim == DimX) {
      nk = l->activity->loc.nx;
      sk = nf;
      line = l->activity->data + nf * nk * loc;
   }
   else {
      nk = l->activity->loc.ny;
      sk = nf * l->activity->loc.nx;
      line = l->activity->data + nf * loc;
   }

   double sum = 0.0;
   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      sum += a;
   }

   float freq = sum / (nk * dt * 0.001);
   fprintf(fp, "t=%6.1f sum=%3d f=%6.1f Hz :", time, (int)sum, freq);

   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      if (a > 0.0) fprintf(fp, "*");
      else         fprintf(fp, " ");
   }

   fprintf(fp, ":\n");
   fflush(fp);

   return 0;
}

} // namespace PV
