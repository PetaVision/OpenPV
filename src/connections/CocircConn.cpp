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

   numFlanks = (int) params->value(name, "numFlanks", numFlanks);
   shift = params->value(name, "flankShift", shift);
   rotate = params->value(name, "rotate", rotate);

   int noPre = pre->clayer->numFeatures;
   noPre = (int) params->value(name, "noPre", noPre);
   assert(noPre > 0);

   int noPost = post->clayer->numFeatures;
   noPost = (int) params->value(name, "noPost", noPost);
   assert(noPost > 0);

   float sigma_cocirc = PI / 2.0;
   sigma_cocirc = params->value(name, "sigma_cocirc", sigma_cocirc);

   float sigma_kurve = 1.0 / sqrt(this->nxp * this->nxp + this->nyp * this->nyp);
   sigma_kurve = params->value(name, "sigma_kurve", sigma_kurve);

   float deltaThetaMax = PI / 2.0;
   deltaThetaMax = params->value(name, "deltaThetaMax", deltaThetaMax);

   float cocirc_self = (pre != post);
   cocirc_self = params->value(name, "cocirc_self", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   float dKv = 1.0; // 1 / minimum radius of curvature
   dKv = params->value(name, "deltaCurvature", dKv);

   for (int kernelIndex = 0; kernelIndex < numPatches; kernelIndex++) {
      int patchIndex = kernelIndexToPatchIndex(kernelIndex);
      cocircCalcWeights(patches[kernelIndex], patchIndex, noPre, noPost, sigma_cocirc,
            sigma_kurve, deltaThetaMax, cocirc_self, dKv, numFlanks, shift, aspect,
            rotate, sigma, r2Max, strength);
   }

   return patches;
}

int CocircConn::cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float deltaThetaMax, float cocirc_self,
      float dKv, int numFlanks, float shift, float aspect, float rotate, float sigma,
      float r2Max, float strength)
{
   pvdata_t * w = wp->data;

   const float min_weight = 0.001; // read in as param
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

   const int kfPre = kPre % pre->clayer->numFeatures;
   const float kurvePre = 0.0 + iKvPre * dKv;
   const float thetaPre = th0 + iThPre * dTh;

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch; kfPost++) {
      int iKvPost = kfPost % nKurvePost;
      int iThPost = kfPost / nKurvePost;

      float kurvePost = 0.0 + iKvPost * dKv;
      float thetaPost = th0 + iThPost * dTh;

      float deltaTheta = fabsf(thetaPre - thetaPost);
      deltaTheta = deltaTheta <= PI / 2.0 ? deltaTheta : PI - deltaTheta;
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
            float dxP = +xDelta * cosf(thetaPre) + yDelta * sinf(thetaPre);
            float dyP = -xDelta * sinf(thetaPre) + yDelta * cosf(thetaPre);

            // include shift to flanks
            float dyP_shift = dyP - shift;
            float dyP_shift2 = dyP + shift;
            float d2 = dxP * dxP + aspect * dyP * aspect * dyP;
            float d2_shift = dxP * dxP + (aspect * (dyP_shift) * aspect * (dyP_shift));
            float d2_shift2 = dxP * dxP + (aspect * (dyP_shift2) * aspect * (dyP_shift2));
            if (d2_shift <= r2Max) {
               gDist += expf(-d2_shift / sigma2);
            }
            if (numFlanks > 1) {
               // include shift in opposite direction
               if (d2_shift2 <= r2Max) {
                  gDist += expf(-d2_shift2 / sigma2);
               }
            }
            if (gDist == 0.0) continue;
            if (d2 == 0) {
               bool sameLoc = (kfPre == kfPost);
               if ((!sameLoc) || (cocirc_self)) {
                  gCocirc = sigma_cocirc > 0 ? expf(-deltaTheta * deltaTheta
                        / sigma_cocirc2) : expf(-deltaTheta * deltaTheta / sigma_cocirc2)
                        - 1.0;
                  if ((nKurvePre > 1) && (nKurvePost > 1)) {
                     gKurvePre = expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
                           / sigma_kurve2);
                  }
               }
               else { // sameLoc && !cocircSelf
                  gCocirc = 0.0;
                  continue;
               }
            }
            else { // d2 > 0

               float atanx2_shift = thetaPre + 2. * atan2f(dyP_shift, dxP); // preferred angle (rad)
               atanx2_shift += 2. * PI;
               atanx2_shift = fmodf(atanx2_shift, PI);
               float chi_shift = fabsf(atanx2_shift - thetaPost); // degrees
               if (chi_shift >= PI / 2.0) {
                  chi_shift = PI - chi_shift;
               }
               if (noPre > 1 && noPost > 1) {
                  gCocirc = sigma_cocirc2 > 0 ? expf(-chi_shift * chi_shift
                        / sigma_cocirc2) : expf(-chi_shift * chi_shift / sigma_cocirc2)
                        - 1.0;
               }

               float cocircKurve_shift = fabsf(2 * dyP_shift) / d2;
               gKurvePre = (nKurvePre > 1) ? expf(-powf(
                     (cocircKurve_shift - fabsf(kurvePre)), 2) / sigma_kurve2) : 1.0;
               gKurvePost =
                     ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0)) ? expf(
                           -powf((cocircKurve_shift - fabsf(kurvePost)), 2) / sigma_kurve2)
                           : 1.0;

               if (numFlanks > 1) {
                  float atanx2_shift2 = thetaPre + 2. * atan2f(dyP_shift2, dxP); // preferred angle (rad)
                  atanx2_shift2 += 2. * PI;
                  atanx2_shift2 = fmodf(atanx2_shift2, PI);
                  float chi_shift2 = fabsf(atanx2_shift2 - thetaPost); // degrees
                  if (chi_shift2 >= PI / 2.0) {
                     chi_shift2 = PI - chi_shift2;
                  }
                  if (noPre > 1 && noPost > 1) {
                     gCocirc += sigma_cocirc2 > 0 ? expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) : expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) - 1.0;
                  }

                  float cocircKurve_shift2 = fabsf(2 * dyP_shift2) / d2;
                  gKurvePre += (nKurvePre > 1) ? expf(-powf((cocircKurve_shift2 - fabsf(
                        kurvePre)), 2) / sigma_kurve2) : 1.0;
                  gKurvePost += ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2
                        > 0)) ? expf(-powf((cocircKurve_shift2 - fabsf(kurvePost)), 2)
                        / sigma_kurve2) : 1.0;
               }
            }
            float weight_tmp = gDist * gKurvePre * gKurvePost * gCocirc;
            if (weight_tmp < min_weight) continue;
            w[iPost * sx + jPost * sy + kfPost * sf] = weight_tmp;

         }
      }
   }

   return 0;

}

} // namespace PV
