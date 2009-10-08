/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointProbe.hpp"
#include <string.h>

namespace PV {

/**
 * @filename
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(const char * filename, int xLoc, int yLoc, int fLoc, const char * msg)
   : PVLayerProbe(filename)
{
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
}

/**
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
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

/**
 * @time
 * @l
 */
int PointProbe::outputState(float time, PVLayer * l)
{
   int nf = l->numFeatures;
   int sf = 1;
   int sx = nf;
   int sy = nf * (l->loc.nx + 2*l->loc.nPad);
   int offset = yLoc * sy + xLoc * sx + fLoc * sf;

   fprintf(fp, "%s t=%6.1f", msg, time);
   fprintf(fp, " G_E=%6.3f", l->G_E[offset]);
   fprintf(fp, " G_I=%6.3f", l->G_I[offset]);
   fprintf(fp, " V=%6.3f", l->V[offset]);
   fprintf(fp, " Vth=%6.3f", l->Vth[offset]);
   fprintf(fp, " a=%3.1f\n", l->activity->data[offset]);

   fflush(fp);

   return 0;
}

} // namespace PV
