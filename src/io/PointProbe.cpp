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
   const float nx = l->loc.nx;
   const float ny = l->loc.ny;
   const float nf = l->numFeatures;

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, l->loc.nPad);

   fprintf(fp, "%s t=%6.1f", msg, time);
   fprintf(fp, " G_E=%6.3f", l->G_E[k]);
   fprintf(fp, " G_I=%6.3f", l->G_I[k]);
   fprintf(fp, " V=%6.3f", l->V[k]);
   fprintf(fp, " Vth=%6.3f", l->Vth[k]);
   fprintf(fp, " a=%3.1f\n", l->activity->data[kex]);

   fflush(fp);

   return 0;
}

} // namespace PV
