/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointProbe.hpp"
#include <string.h>

namespace PV {

PointProbe::PointProbe(const char * filename, int xLoc, int yLoc, int fLoc, const char * msg)
   : PVLayerProbe(filename)
{
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
}

PointProbe::PointProbe(int xLoc, int yLoc, int fLoc, const char * msg)
   : PVLayerProbe()
{
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
}

PointProbe::~PointProbe()
{
   free(msg);
}

int PointProbe::outputState(float time, PVLayer * l)
{
   int nf = l->numFeatures;
   int sf = 1;
   int sx = nf;
   int sy = nf * l->loc.nx;
   int offset = yLoc * sy + xLoc * sx + fLoc * sf;

   fprintf(fp, "%s t=%6.1f", msg, time);
   fprintf(fp, " G_E=%6.4f", l->G_E[offset]);
   fprintf(fp, " G_I=%6.4f", l->G_I[offset]);
   fprintf(fp, " V=%6.4f", l->V[offset]);
   fprintf(fp, " Vth=%6.4f", l->Vth[offset]);
   fprintf(fp, " a=%3.1f\n", l->activity->data[offset]);

   fflush(fp);

   return 0;
}

} // namespace PV
