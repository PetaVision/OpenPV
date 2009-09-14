/*
 * GaborConn.cpp
 *
 *  Created on: Jan 12, 2009
 *      Author: rasmussn
 */

#include "GaborConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

GaborConn::GaborConn(const char * name,
                     HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
         : HyPerConn(name, hc, pre, post, channel, PROTECTED_NUMBER)
{
   this->numAxonalArborLists = 1;
   initialize();
   hc->addConnection(this);
}

int GaborConn::initializeWeights(const char * filename)
{
   float aspect = 4.0;
   float sigma  = 2.0;
   float rMax   = 8.0;
   float lambda = sigma/0.8;    // gabor wavelength
   float strength = 1.0;

   PVParams * params = parent->parameters();

   aspect = params->value(name, "aspect");
   sigma  = params->value(name, "sigma");
   rMax   = params->value(name, "rMax");
   lambda = params->value(name, "lambda");

   if (params->present(name, "strength")) {
      strength = params->value(name, "strength");
   }

   float r2Max = rMax * rMax;

   const int borderId = 0;
   const int numPatches = numberOfWeightPatches(borderId);
   for (int i = 0; i < numPatches; i++) {
      int xScale = post->clayer->xScale - pre->clayer->xScale;
      int yScale = post->clayer->xScale - pre->clayer->yScale;
      gaborWeights(wPatches[borderId][i], xScale, yScale, aspect, sigma, r2Max, lambda, strength);
   }

   return 0;
}

int GaborConn::gaborWeights(PVPatch * wp, int xScale, int yScale,
                            float aspect, float sigma, float r2Max, float lambda, float strength)
{
   PVParams * params = parent->parameters();

   float rotate = 1.0;
   float invert = 0.0;
   if (params->present(name, "rotate")) rotate = params->value(name, "rotate");
   if (params->present(name, "invert")) invert = params->value(name, "invert");

   pvdata_t * w = wp->data;

   const float phi = 0.0;  // phase

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   const float dx = powf(2, xScale);
   const float dy = powf(2, yScale);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   const float x0 = -(nx/2.0 - 0.5) * dx;
   const float y0 = +(ny/2.0 - 0.5) * dy;

   const float dth = PI/nf;
   const float th0 = rotate*dth/2.0;

   for (int f = 0; f < nf; f++) {
      float th = th0 + f * dth;
      for (int j = 0; j < ny; j++) {
         float yp = y0 - j * dy;    // pixel coordinate
         for (int i = 0; i < nx; i++) {
            float xp  = x0 + i*dx;  // pixel coordinate

            // rotate the reference frame by th ((x,y) is center of patch (0,0))
            float u1 = + (0.0 - xp) * cos(th) + (0.0 - yp) * sin(th);
            float u2 = - (0.0 - xp) * sin(th) + (0.0 - yp) * cos(th);

            float factor = cos(2.0*PI*u2/lambda + phi);
            if (fabs(u2/lambda) > 3.0/4.0) factor = 0.0;  // phase < 3*PI/2 (no second positive band)
            float d2 = u1 * u1 + (aspect*u2 * aspect*u2);
            float wt = factor * expf(-d2 / (2.0*sigma*sigma));

#ifdef DEBUG_OUTPUT
            if (j == 0) printf("x=%f fac=%f w=%f\n", xp, factor, wt);
#endif
            if (xp*xp + yp*yp > r2Max) {
               w[i*sx + j*sy + f*sf] = 0.0;
            }
            else {
               if (invert) wt *= -1.0;
               if (wt < 0.0) wt = 0.0;       // clip negative values
               w[i*sx + j*sy + f*sf] = wt;
            }
         }
      }
   }

   // normalize
   for (int f = 0; f < nf; f++) {
      float sum = 0;
      for (int i = 0; i < nx*ny; i++) {
         if (w[f + i*nf] > 0.0) sum += w[f + i*nf];
      }

      if (sum == 0.0) continue;  // all weights == zero is ok

      float factor = strength/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

} // namespace PV
