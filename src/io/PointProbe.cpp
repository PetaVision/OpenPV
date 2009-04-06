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

   fprintf(fd, "%s t=%4d", msg, (int)time);
   fprintf(fd, " G_E=%1.4f", l->G_E[offset]);
   fprintf(fd, " G_I=%1.4f", l->G_I[offset]);
   fprintf(fd, " V=%1.4f", l->V[offset]);
   fprintf(fd, " Vth=%1.4f", l->Vth[offset]);
   fprintf(fd, " a=%1.1f\n", l->activity->data[offset]);

   fflush(fd);

   return 0;
}

} // namespace PV
