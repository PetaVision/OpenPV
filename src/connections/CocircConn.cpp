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
   assert(noPre <= pre->clayer->numFeatures);

   int noPost = post->clayer->numFeatures;
   noPost = (int) params->value(name, "noPost", noPost);
   assert(noPost > 0);
   assert(noPost <= post->clayer->numFeatures);

   float sigma_cocirc = PI / 2.0;
   sigma_cocirc = params->value(name, "sigmaCocirc", sigma_cocirc);

   float sigma_kurve = 1.0; // fraction of delta_radius_curvature
   sigma_kurve = params->value(name, "sigmaKurve", sigma_kurve);

   // sigma_chord = % of PI * R, where R == radius of curvature (1/curvature)
   float sigma_chord = 0.5;
   sigma_chord = params->value(name, "sigmaChord", sigma_chord);

   float delta_theta_max = PI / 2.0;
   delta_theta_max = params->value(name, "deltaThetaMax", delta_theta_max);

   float cocirc_self = (pre != post);
   cocirc_self = params->value(name, "cocircSelf", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   float delta_radius_curvature = 1.0; // 1 = minimum radius of curvature
   delta_radius_curvature = params->value(name, "deltaRadiusCurvature",
         delta_radius_curvature);

   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      cocircCalcWeights(patches[patchIndex], patchIndex, noPre, noPost, sigma_cocirc,
            sigma_kurve, sigma_chord, delta_theta_max, cocirc_self,
            delta_radius_curvature, numFlanks, shift, aspect, rotate, sigma, r2Max,
            strength);
   }

   return patches;
}

int CocircConn::cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
      float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
      float aspect, float rotate, float sigma, float r2Max, float strength)
{
   pvdata_t * w = wp->data;

   const float min_weight = 0.0f; // read in as param
   const float sigma2 = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;

   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = (int) wp->nf;
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   // get strides of (potentially shrunken) patch
   const int sx = (int) wp->sx;
   assert(sx == nfPatch);
   const int sy = (int) wp->sy; // no assert here because patch may be shrunken
   const int sf = (int) wp->sf;
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   PVPatch * wp_tmp;
   wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   pvdata_t * w_tmp = wp_tmp->data;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = wp_tmp->nx;
   const int nyPatch_tmp = wp_tmp->ny;
   const int nfPatch_tmp = wp_tmp->nf;
   int kxKernelIndex;
   int kyKerneIndex;
   int kfKernelIndex;
   this->patchIndexToKernelIndex(kPre, &kxKernelIndex, &kyKerneIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKerneIndex;
   //   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp = wp_tmp->sx;
   assert(sx_tmp == wp_tmp->nf);
   const int sy_tmp = wp_tmp->sy;
   assert(sy_tmp == wp_tmp->nf * wp_tmp->nx);
   const int sf_tmp = wp_tmp->sf;
   assert(sf_tmp == 1);

   // get distances to nearest neighbor in post synaptic layer
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre_tmp, pre->getXScale(), post->getXScale(), &xDistNNPreUnits,
         &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kyPre_tmp, pre->getYScale(), post->getYScale(), &yDistNNPreUnits,
         &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre_tmp, pre->getXScale(), post->getXScale());
   kyNN = nearby_neighbor(kyPre_tmp, pre->getYScale(), post->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, pre->getYScale(), post->getYScale());

   // get distance to patch head
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale = xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits
         / xDistNNPostUnits;
   float xDistHeadPreUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale = yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits
         / yDistNNPostUnits;
   float yDistHeadPreUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;

   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, post->clayer->xScale);
   const float dyPost = powf(2, post->clayer->yScale);

   //const int kfPre = kPre % pre->clayer->numFeatures;
   const int kfPre = featureIndex(kPre, pre->clayer->loc.nx, pre->clayer->loc.ny,
         pre->clayer->numFeatures);

   bool POS_KURVE_FLAG = false; //  handle pos and neg curvature separately
   bool SADDLE_FLAG  = false; // handle saddle points separately
   const int nKurvePre = pre->clayer->numFeatures / noPre;
   const int nKurvePost = post->clayer->numFeatures / noPost;
   const float dTh = PI / nfPatch;
   const float th0 = rotate * dTh / 2.0;
   const int iThPre = kPre / nKurvePre;
   const float thetaPre = th0 + iThPre * dTh;

   int iKvPre = kfPre % nKurvePre;
   bool iPosKurvePre = false;
   bool iSaddlePre = false;
   float radKurvPre = delta_radius_curvature + iKvPre * delta_radius_curvature;
   float kurvePre = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   int iKvPreAdj = iKvPre;
   if (POS_KURVE_FLAG) {
      assert(nKurvePre >= 2);
      iPosKurvePre = iKvPre >= (int) (nKurvePre / 2);
      if (SADDLE_FLAG) {
         assert(nKurvePre >= 4);
         iSaddlePre = (iKvPre % 2 == 0) ? 0 : 1;
         iKvPreAdj = ((iKvPre % (nKurvePre / 2)) / 2);}
      else { // SADDLE_FLAG
         iKvPreAdj = (iKvPre % (nKurvePre/2));}
   } // POS_KURVE_FLAG
   radKurvPre = delta_radius_curvature + iKvPreAdj * delta_radius_curvature;
   kurvePre = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   float sigma_kurve_pre = sigma_kurve * radKurvPre;
   float sigma_kurve_pre2 = 2 * sigma_kurve_pre * sigma_kurve_pre;
   sigma_chord *= PI * radKurvPre;
   float sigma_chord2 = 2.0 * sigma_chord * sigma_chord;

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch_tmp; kfPost++) {
      int iThPost = kfPost / nKurvePost;
      float thetaPost = th0 + iThPost * dTh;

      int iKvPost = kfPost % nKurvePost;
      bool iPosKurvePost = false;
      bool iSaddlePost = false;
      float radKurvPost = delta_radius_curvature + iKvPost * delta_radius_curvature;
      float kurvePost = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      int iKvPostAdj = iKvPost;
      if (POS_KURVE_FLAG) {
         assert(nKurvePost >= 2);
         iPosKurvePost = iKvPost >= (int) (nKurvePost / 2);
         if (SADDLE_FLAG) {
            assert(nKurvePost >= 4);
            iSaddlePost = (iKvPost % 2 == 0) ? 0 : 1;
            iKvPostAdj = ((iKvPost % (nKurvePost / 2)) / 2);
         }
         else { // SADDLE_FLAG
            iKvPostAdj = (iKvPost % (nKurvePost / 2));
         }
      } // POS_KURVE_FLAG
      radKurvPost = delta_radius_curvature + iKvPostAdj * delta_radius_curvature;
      kurvePost = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      float sigma_kurve_post = sigma_kurve * radKurvPost;
      float sigma_kurve_post2 = 2 * sigma_kurve_post * sigma_kurve_post;

      float deltaTheta = fabsf(thetaPre - thetaPost);
      deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
      if (deltaTheta > delta_theta_max) {
         continue;
      }

      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = (yDistHeadPreUnits + jPost * dyPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = (xDistHeadPreUnits + iPost * dxPost);

            float gDist = 0.0;
            float gChord = 1.0;
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
                           / 2 * (sigma_kurve_pre * sigma_kurve_pre + sigma_kurve_post
                           * sigma_kurve_post));
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
               atanx2_shift = fabsf(atanx2_shift - thetaPost);
               float chi_shift = atanx2_shift; //fabsf(atanx2_shift - thetaPost); // radians
               if (chi_shift >= PI / 2.0) {
                  chi_shift = PI - chi_shift;
               }
               if (noPre > 1 && noPost > 1) {
                  gCocirc = sigma_cocirc2 > 0 ? expf(-chi_shift * chi_shift
                        / sigma_cocirc2) : expf(-chi_shift * chi_shift / sigma_cocirc2)
                        - 1.0;
               }

               // compute curvature of cocircular contour
               float cocircKurve_shift = d2_shift > 0 ? fabsf(2 * dyP_shift) / d2_shift
                     : 0.0f;
               if (POS_KURVE_FLAG) {
                  if (SADDLE_FLAG) {
                     if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift > 0)) {
                        continue;
                     }
                     if ((iPosKurvePre) && (iSaddlePre)
                           && (((dyP_shift > 0) && (dxP < 0)) || ((dyP_shift > 0) && (dxP
                                 < 0)))) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (iSaddlePre) && (((dyP_shift > 0)
                           && (dxP > 0)) || ((dyP_shift < 0) && (dxP < 0)))) {
                        continue;
                     }
                  }
                  else { //SADDLE_FLAG
                     if ((iPosKurvePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (dyP_shift > 0)) {
                        continue;
                     }
                  }
               } // POS_KURVE_FLAG
               gKurvePre = (nKurvePre > 1) ? expf(-powf((cocircKurve_shift - fabsf(
                     kurvePre)), 2) / sigma_kurve_pre2) : 1.0;
               gKurvePost
                     = ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0)) ? expf(
                           -powf((cocircKurve_shift - fabsf(kurvePost)), 2)
                                 / sigma_kurve_post2)
                           : 1.0;

               // compute distance along contour
               float d_chord_shift = (cocircKurve_shift != 0.0f) ? atanx2_shift
                     / cocircKurve_shift : sqrt(d2_shift);
               gChord = (nKurvePre > 1) ? expf(-powf(d_chord_shift, 2) / sigma_chord2)
                     : 1.0;

               if (numFlanks > 1) {
                  float atanx2_shift2 = thetaPre + 2. * atan2f(dyP_shift2, dxP); // preferred angle (rad)
                  atanx2_shift2 += 2. * PI;
                  atanx2_shift2 = fmodf(atanx2_shift2, PI);
                  atanx2_shift2 = fabsf(atanx2_shift2 - thetaPost);
                  float chi_shift2 = atanx2_shift2; //fabsf(atanx2_shift2 - thetaPost); // radians
                  if (chi_shift2 >= PI / 2.0) {
                     chi_shift2 = PI - chi_shift2;
                  }
                  if (noPre > 1 && noPost > 1) {
                     gCocirc += sigma_cocirc2 > 0 ? expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) : expf(-chi_shift2 * chi_shift2
                           / sigma_cocirc2) - 1.0;
                  }

                  float cocircKurve_shift2 = d2_shift2 > 0 ? fabsf(2 * dyP_shift2)
                        / d2_shift2 : 0.0f;
                  if (POS_KURVE_FLAG) {
                     if (SADDLE_FLAG) {
                        if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                        if ((iPosKurvePre) && (iSaddlePre) && (((dyP_shift2 > 0) && (dxP
                              < 0)) || ((dyP_shift2 > 0) && (dxP < 0)))) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (iSaddlePre) && (((dyP_shift2 > 0) && (dxP
                              > 0)) || ((dyP_shift2 < 0) && (dxP < 0)))) {
                           continue;
                        }
                     }
                     else { //SADDLE_FLAG
                        if ((iPosKurvePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                     } // SADDLE_FLAG
                  } // POS_KURVE_FLAG
                  gKurvePre += (nKurvePre > 1) ? expf(-powf((cocircKurve_shift2 - fabsf(
                        kurvePre)), 2) / sigma_kurve_pre2) : 1.0;
                  gKurvePost += ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2
                        > 0)) ? expf(-powf((cocircKurve_shift2 - fabsf(kurvePost)), 2)
                        / sigma_kurve_post2) : 1.0;

                  float d_chord_shift2 = cocircKurve_shift2 != 0.0f ? atanx2_shift2
                        / cocircKurve_shift2 : sqrt(d2_shift2);
                  gChord += (nKurvePre > 1) ? expf(-powf(d_chord_shift2, 2) / sigma_chord2)
                        : 1.0;

               }
            }
            float weight_tmp = gDist * gKurvePre * gKurvePost * gCocirc;
            if (weight_tmp < min_weight) continue;
            w_tmp[iPost * sx_tmp + jPost * sy_tmp + kfPost * sf_tmp] = weight_tmp;

         }
      }
   }

   // copy weights from full sized temporary patch to (possibly shrunken) patch
   w = wp->data;
   pvdata_t * data_head = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset = w - data_head;
   w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }

   free(wp_tmp);
   return 0;

}

} // namespace PV
