/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
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
   : LayerProbe(filename)
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
   : LayerProbe()
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
int PointProbe::outputState(float time, HyPerLayer * l)
{
   const PVLayer * clayer = l->clayer;

   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   const float nf = clayer->numFeatures;

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, clayer->loc.nPad);

   fprintf(fp, "%s t=%6.1f", msg, time);
   fprintf(fp, " G_E=%6.3f", clayer->G_E[k]);
   fprintf(fp, " G_I=%6.3f", clayer->G_I[k]);
   fprintf(fp, " V=%6.3f",   clayer->V[k]);
   fprintf(fp, " Vth=%6.3f", clayer->Vth[k]);
   fprintf(fp, " a=%3.1f\n", clayer->activity->data[kex]);

   fflush(fp);

   return 0;
}

} // namespace PV
