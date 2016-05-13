/*
 * GaborConn.cpp
 *
 *  Created on: Jan 12, 2009
 *      Author: rasmussn
 */

#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "GaborWeight" in the params file

#include "GaborConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

GaborConn::GaborConn(const char * name,
                     HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, NULL);
}

PVPatch ** GaborConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   return initializeGaborWeights(patches, numPatches);
}

PVPatch ** GaborConn::initializeGaborWeights(PVPatch ** patches, int numPatches)
{
#ifdef TESTING
	   PVPatch * wp = kernelPatches[0];
	   pvdata_t * w = wp->data;

	   const int nxp = wp->nx;
	   const int nyp = wp->ny;
	   const int nfp = wp->nf;
	   if (nxp * nyp * nfp == 0) {
	      return 0; // reduced patch size is zero
	   }

	   for (int k = 0; k < nxp*nyp*nfp; k++) {
		   w[k] = 1.0f;
	   }
#else

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->xScale - pre->clayer->yScale;

   PVParams * params = parent->parameters();

   float aspect = 4.0;
   float sigma  = 2.0;
   float rMax   = 8.0;
   float lambda = sigma/0.8;    // gabor wavelength
   float strength = 1.0;
   float phi = 0;

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   lambda   = params->value(name, "lambda", lambda);
   strength = params->value(name, "strength", strength);
   phi = params->value(name, "phi", phi);

   float r2Max = rMax * rMax;

   for (int kernelIndex = 0; kernelIndex < numPatches; kernelIndex++) {
      // TODO - change parameters based on kernelIndex (i.e., change orientation)
      gaborWeights(patches[kernelIndex], xScale, yScale, aspect, sigma, r2Max, lambda, strength, phi);
   }
#endif
   return patches;
}

int GaborConn::gaborWeights(PVPatch * wp, int xScale, int yScale,
                            float aspect, float sigma, float r2Max, float lambda, float strength, float phi)
{
   PVParams * params = parent->parameters();

   float rotate = 1.0;
   float invert = 0.0;
   if (params->present(name, "rotate")) rotate = params->value(name, "rotate");
   if (params->present(name, "invert")) invert = params->value(name, "invert");

   pvdata_t * w = wp->data;

   //const float phi = 3.1416;  // phase

   const int nx = wp->nx;
   const int ny = wp->ny;
   const int nf = wp->nf;

   const int sx = wp->sx;  assert(sx == nf);
   const int sy = wp->sy;  assert(sy == nf*nx);
   const int sf = wp->sf;  assert(sf == 1);

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

} // Namespace PV
#endif
