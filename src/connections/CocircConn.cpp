/*
 * CocircConn.cpp
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#include "CocircConn.hpp"
#include "../io/io.h"
#include "../utils/conversions.h"
#include <assert.h>
#include <string.h>

namespace PV {

CocircConn::CocircConn()
{
   printf("CocircConn::CocircConn: running default constructor\n");
   initialize_base();
}

CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel);
}

CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL); // use default channel
}

// provide filename or set to NULL
CocircConn::CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
}

PVPatch ** CocircConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   return initializeCocircWeights(patches, numPatches);
}

PVPatch ** CocircConn::initializeCocircWeights(PVPatch ** patches, int numPatches)
{
   PVParams * params = parent->parameters();
   float aspect = 1.0; // circular (not line oriented)
   float sigma = 0.8;
   float rMax = 1.4;
   float strength = 1.0;

   aspect = params->value(name, "aspect", aspect);
   sigma = params->value(name, "sigma", sigma);
   rMax = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift = 0.0f;
   float rotate = 0.0f; // rotate so that axis isn't aligned

   numFlanks = params->value(name, "numFlanks", numFlanks);
   shift = params->value(name, "flankShift", shift);
   rotate = params->value(name, "rotate", rotate);

   int noPre = pre->clayer->numFeatures;
   noPre = params->value(name, "noPre", noPre);
   assert(noPre> 0);

   int noPost = post->clayer->numFeatures;
   noPost = params->value(name, "noPost", noPost);
   assert(noPost> 0);

   float sigma_cocirc = PI / 2.0;
   sigma_cocirc = params->value(name, "sigma_cocirc", sigma_cocirc);

   float sigma_kurve = 1.0 / sqrt( this->nxp * this->nxp + this->nyp * this->nyp );
   sigma_kurve = params->value(name, "sigma_kurve", sigma_kurve);

   float deltaThetaMax = PI / 2.0;
   deltaThetaMax = params->value(name, "deltaThetaMax", deltaThetaMax);

   float cocirc_self = pre != post;
   cocirc_self = params->value(name, "cocirc_self", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   float dKv = 1.0; // 1 / minimum radius of curvature
   dKv = params->value(name, "deltaCurvature", dKv);

   for (int kernelIndex = 0; kernelIndex < numPatches; kernelIndex++) {
      int patchIndex = kernelIndexToPatchIndex(kernelIndex);
      cocircCalcWeights(patches[kernelIndex], patchIndex, noPre, noPost, sigma_cocirc, sigma_kurve,
            deltaThetaMax, cocirc_self, dKv, numFlanks, shift, aspect, rotate, sigma,
            r2Max, strength);
   }

   return patches;
}

int CocircConn::cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float deltaThetaMax, float cocirc_self,
      float dKv, int numFlanks, float shift, float aspect, float rotate, float sigma,
      float r2Max, float strength)
{
#define OLD_COCIRCCALCWEIGHTS 0
#if OLD_COCIRCCALCWEIGHTS == 0  // use new method
   pvdata_t * w = wp->data;

   const float min_weight = 0.001;  // read in as param
   const float sigma2 = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;
   const float sigma_kurve2 = 2 * sigma_kurve * sigma_kurve;

   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = (int) wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   float xPreGlobal = 0.0;
   float yPreGlobal = 0.0;
   float xPatchHeadGlobal = 0.0;
   float yPatchHeadGlobal = 0.0;

   posPatchHead(kPre, pre->clayer->xScale, pre->clayer->yScale, pre->clayer->loc,
         &xPreGlobal, &yPreGlobal, post->clayer->xScale, post->clayer->yScale,
         post->clayer->loc, wp, &xPatchHeadGlobal, &yPatchHeadGlobal);

   // ready to compute weights
   const int sx = (int) wp->sx;
   assert(sx == nfPatch);
   const int sy = (int) wp->sy; // no assert here because patch may be shrunken
   const int sf = (int) wp->sf;
   assert(sf == 1);

   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, post->clayer->xScale);
   const float dyPost = powf(2, post->clayer->yScale);

   const float dTh = PI / nfPatch;
   const float th0 = rotate * dTh / 2.0;

   const int nKurvePre = pre->clayer->numFeatures / noPre;
   const int nKurvePost = pre->clayer->numFeatures / noPost;

   const int iKvPre = kPre % nKurvePre;
   const int iThPre = kPre / nKurvePre;

   const float kurvePre = 0.0 + iKvPre * dKv;
   const float thetaPre = th0 + iThPre * dTh;

   // loop over all post synaptic neurons in patch
   for (int f = 0; f < nfPatch; f++) {
      int iKvPost = f % nKurvePost;
      int iThPost = f / nKurvePost;

      float kurvePost = 0.0 + iKvPost * dKv;
      float thetaPost = th0 + iThPost * dTh;

      float deltaTheta = fabs(thetaPre - thetaPost);
      deltaTheta = deltaTheta <= PI/2.0 ? deltaTheta : PI - deltaTheta;
      if (deltaTheta > deltaThetaMax) {
         continue;
      }

      float xDelta = 0.0;
      float yDelta = 0.0;
     for (int jPost = 0; jPost < nyPatch; jPost++) {
        yDelta = (yPatchHeadGlobal + jPost * dyPost) - yPreGlobal;
         for (int iPost = 0; iPost < nxPatch; iPost++) {
            xDelta = (xPatchHeadGlobal + iPost * dxPost) - xPreGlobal;

            float gDist = 0.0;
            float gCocirc = 1.0;
            float gKurvePre = 1.0;
            float gKurvePost = 1.0;

                   // rotate the reference frame by th
            float dxP = +xDelta * cos(thetaPre) + yDelta * sin(thetaPre);
            float dyP = -xDelta * sin(thetaPre) + yDelta * cos(thetaPre);

            // include shift to flanks
            float d2 = dxP * dxP + (aspect * (dyP - shift) * aspect * (dyP - shift));
            float d2_2 = (numFlanks > 1) ? dxP * dxP + (aspect * (dyP + shift) * aspect * (dyP + shift)) : 2*r2Max;
            if (d2 <= r2Max) {
               gDist += expf(-d2 / sigma2);
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               if (d2_2 <= r2Max) {
                  gDist += expf(-d2_2 / sigma2);
               }
            }
            if (gDist == 0.0) continue;
            if (((d2 == 0) || (d2_2 == 0)) && (cocirc_self)) {
               gCocirc
                     = sigma_cocirc > 0 ? expf(-deltaTheta * deltaTheta / sigma_cocirc2)
                           : expf(-deltaTheta * deltaTheta / sigma_cocirc2) - 1.0;
               if ((nKurvePre > 1) && (nKurvePost > 1)) {
                  gKurvePre = expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
                        / sigma_kurve2);
               }
            }
            else { // d2 > 0

               float atanx2 = thetaPre + 2. * atan2f(dyP, dxP); // preferred angle (rad)
               atanx2 += 2. * PI;
               atanx2 = fmod(atanx2, PI );
               float chi = fabs(atanx2 - thetaPost); // degrees
               if (chi >= PI/2.0) {
                  chi = PI - chi;
               }
               if (noPre > 1 && noPost > 1) {
                  gCocirc = sigma_cocirc2 > 0 ? expf(-chi * chi / sigma_cocirc2) : expf(
                        -chi * chi / sigma_cocirc2) - 1.0;
               }

               float cocircKurve = fabs(2 * dyP) / d2;
               gKurvePre = (nKurvePre > 1) ? exp(-pow((cocircKurve - fabs(kurvePre)), 2)
                     / sigma_kurve2) : 1.0;
               gKurvePost
                     = ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0)) ? exp(
                           -pow((cocircKurve - fabs(kurvePost)), 2) / sigma_kurve2)
                           : 1.0;
            }
            float weight_tmp = gDist * gKurvePre * gKurvePost * gCocirc;
            if (weight_tmp < min_weight) continue;
            w[iPost * sx + jPost * sy + f * sf] = weight_tmp;

         }
      }
   }

   return 0;

#else  // use old method
   float gDist = 0.0;

   pvdata_t * w = wp->data;

   // get parameters

   // PVParams * params = parent->parameters();

   const int nfPre = pre->clayer->numFeatures;
   const int nfPost = post->clayer->numFeatures;

   const float sigma2 = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;
   const float sigma_kurve2 = 2 * sigma_kurve * sigma_kurve;

   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = (int) wp->nf;

   // strides
   const int sx = (int) wp->sx;
   assert(sx == nfPatch);
   const int sy = (int) wp->sy;
   assert(sy == nfPatch*nxPatch);
   const int sf = (int) wp->sf;
   assert(sf == 1);

   const float dxPost = powf(2, xScale);
   const float dyPost = powf(2, yScale);

   // get 3D index of pre cell
   const int kxPre = (int) kxPos(kPre, nxPre, nyPre, nfPre);
   const int kyPre = (int) kyPos(kPre, nxPre, nyPre, nfPre);
   const int kfPre = (int) featureIndex(kPre, nxPre, nyPre, nfPre);

   // pre x,y location
   //
   const int xPreScaleFac = pow(2, pre->clayer->xScale);
   const int yPreScaleFac = pow(2, pre->clayer->yScale);
   const float xPrePos = ( xPreScaleFac / 2.0 ) + kxPre * xPreScaleFac;
   const float yPrePos = ( yPreScaleFac / 2.0 ) + kyPre * yPreScaleFac;

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   const float x0 = -(nxPatch / 2.0 - 0.5) * dxPost;
   const float y0 = +(nyPatch / 2.0 - 0.5) * dyPost;

   const int nKurvePre = nfPre / noPre;
   const int nKurvePost = nfPost / noPost;

   const float dTh = PI / noPost;
   const float th0 = rotate * dTh / 2.0;

   const int iKvPre = kPre % nKurvePre;
   const int iThPre = kPre / nKurvePre;

   const float kurvePre = 0.0 + iKvPre * dKv;
   const float thetaPre = th0 + iThPre * dTh;

   // loop over all post synaptic neurons in patch
   for (int f = 0; f < nfPatch; f++) {
      int iKvPost = f % nKurvePost;
      int iThPost = f / nKurvePost;

      float kurvePost = 0.0 + iKvPost * dKv;
      float thetaPost = th0 + iThPost * dTh;

      float deltaTheta = RAD_TO_DEG * fabs(thetaPre - thetaPost);
      deltaTheta = deltaTheta <= 90. ? deltaTheta : 180. - deltaTheta;
      if (deltaTheta> deltaThetaMax) {
         continue;
      }

      float gCocirc = 1.0;
      float gKurvePre = 1.0;
      float gKurvePost = 1.0;

      for (int j = 0; j < nyPatch; j++) {
         float y = y0 - j * dyPost;
         for (int i = 0; i < nxPatch; i++) {
            float x = x0 + i * dxPost;
            float d2 = x * x + y * y;
            if (d2> r2Max) continue; // || (d2 < r2Min)

            gDist = expf(-d2 / sigma2);

            if (d2 == 0 && (cocirc_self == 0)) {
               // TODO - why calculate anything else
               gDist = 0.0;
               gCocirc
               = sigma_cocirc> 0 ? expf(-deltaTheta * deltaTheta / sigma_cocirc2)
               : expf(-deltaTheta * deltaTheta / sigma_cocirc2) - 1.0;
               if ((nKurvePre> 1) && (nKurvePost> 1)) {
                  gKurvePre = expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
                        / sigma_kurve2);
               }
            }
            else {
               float dxP = +x * cos(thetaPre) + y * sin(thetaPre);
               float dyP = -x * sin(thetaPre) + y * cos(thetaPre);

               // The first version implements traditional association field
               // of Field, Li, Zucker, etc. It looks like a bowtie for most
               // orientations, although orthogonal angles are supported at
               // 45 degrees in all four quadrants.

               float atanx2 = thetaPre + 2. * atan2f(dyP, dxP); // preferred angle (rad)
               atanx2 += 2. * PI;
               atanx2 = fmod(atanx2, PI );
               float chi = RAD_TO_DEG * fabs(atanx2 - thetaPost); // degrees
               if (chi >= 90.) {
                  chi = 180. - chi;
               }
               if (noPre> 1 && noPost> 1) {
                  gCocirc = sigma_cocirc2> 0 ? expf(-chi * chi / sigma_cocirc2) : expf(
                        -chi * chi / sigma_cocirc2) - 1.0;
               }

               float cocircKurve = fabs(2 * dyP) / d2;
               gKurvePre = (nKurvePre> 1) ? exp(-pow((cocircKurve - fabs(kurvePre)), 2)
                     / sigma_kurve2) : 1.0;
               gKurvePost
               = ((nKurvePre> 1) && (nKurvePost> 1) && (sigma_cocirc2> 0)) ? exp(
                     -pow((cocircKurve - fabs(kurvePost)), 2) / sigma_kurve2)
               : 1.0;
            }

            w[i * sx + j * sy + f * sf] = gDist * gKurvePre * gKurvePost * gCocirc;

         }
      }
   }

   // normalize
   for (int f = 0; f < nfPatch; f++) {
      float sum = 0;
      for (int i = 0; i < nxPatch * nyPatch; i++)
      sum += w[f + i * nfPatch];

      if (sum == 0.0) continue; // all weights == zero is ok

      float factor = strength / sum;
      for (int i = 0; i < nxPatch * nyPatch; i++)
      w[f + i * nfPatch] *= factor;
   }

   return 0;
#endif
}

} // namespace PV
