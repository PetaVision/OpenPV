/*
 * SharedConnDebugInitWeights.cpp
 *
 *  Created on: Aug 22, 2011
 *      Author: kpeterson
 */

#include "SharedConnDebugInitWeights.hpp"
#include "SharedWeightsTrue.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

SharedConnDebugInitWeights::SharedConnDebugInitWeights() {}

SharedConnDebugInitWeights::SharedConnDebugInitWeights(const char *name, HyPerCol *hc)
      : HyPerConn() {
   SharedConnDebugInitWeights::initialize(name, hc);
}

SharedConnDebugInitWeights::~SharedConnDebugInitWeights() {}

int SharedConnDebugInitWeights::initialize(const char *name, HyPerCol *hc) {
   HyPerConn::initialize(name, hc);
   return PV_SUCCESS;
}

SharedWeights *SharedConnDebugInitWeights::createSharedWeights() {
   return new SharedWeightsTrue(name, parent);
}

int SharedConnDebugInitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_weightInitType(ioFlag);
   return status;
}

void SharedConnDebugInitWeights::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "weightInitType", &mWeightInitTypeString, NULL, true /*warnIfAbsent*/);
   FatalIf(
         mWeightInitTypeString == nullptr or mWeightInitTypeString[0] == '\0',
         "%s must set weightInitType.\n",
         getDescription_c());
}

Response::Status SharedConnDebugInitWeights::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return HyPerConn::communicateInitInfo(message);
}

Response::Status SharedConnDebugInitWeights::initializeState() {
   FatalIf(
         mWeightsPair->getPreWeights() == nullptr,
         "SharedConnDebugInitWeights::initializeState called with no presynaptic weights.\n");
   FatalIf(
         mWeightInitTypeString == nullptr or mWeightInitTypeString[0] == '\0',
         "SharedConnDebugInitWeights did not set weightInitTypeString.\n");
   int numKernelPatches = getNumDataPatches();
   int numArbors        = mWeightsPair->getPreWeights()->getNumArbors();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      float *arborStart = getWeightsDataStart(arbor);
      if (!strcmp(mWeightInitTypeString, "CoCircWeight")) {
         initializeCocircWeights(arborStart, numKernelPatches);
      }
      else if (!strcmp(mWeightInitTypeString, "SmartWeight")) {
         initializeSmartWeights(arborStart, numKernelPatches);
      }
      else if (!strcmp(mWeightInitTypeString, "GaborWeight")) {
         initializeGaborWeights(arborStart, numKernelPatches);
      }
      else if (!strcmp(mWeightInitTypeString, "Gauss2DWeight")) {
         initializeGaussian2DWeights(arborStart, numKernelPatches);
      }
   }
   return Response::SUCCESS;
}

void SharedConnDebugInitWeights::initializeSmartWeights(float *dataStart, int numPatches) {
   int overallPatchSize = mWeightsPair->getPreWeights()->getPatchSizeOverall();
   for (int k = 0; k < numPatches; k++) {
      smartWeights(&dataStart[k * overallPatchSize], dataIndexToUnitCellIndex(k));
   }
}

void SharedConnDebugInitWeights::smartWeights(float *dataStart, int k) {
   float *w = dataStart;

   const int nxp = getPatchSizeX();
   const int nyp = getPatchSizeY();
   const int nfp = getPatchSizeF();

   const int sxp = getPatchStrideX();
   const int syp = getPatchStrideY();
   const int sfp = getPatchStrideF();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = k;
         }
      }
   }
}

void SharedConnDebugInitWeights::initializeCocircWeights(float *dataStart, int numPatches) {
   PVParams *params = parent->parameters();
   float aspect     = 1.0f; // circular (not line oriented)
   float sigma      = 0.8f;
   float rMax       = 1.4f;
   float strength   = 1.0f;

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift   = 0.0f;
   float rotate  = 0.0f; // rotate so that axis isn't aligned

   numFlanks = (int)params->value(name, "numFlanks", numFlanks);
   shift     = params->value(name, "flankShift", shift);
   rotate    = params->value(name, "rotate", rotate);

   int noPre = getPre()->getLayerLoc()->nf;
   noPre     = (int)params->value(name, "noPre", noPre);
   FatalIf(!(noPre > 0), "Test failed.\n");
   FatalIf(!(noPre <= getPre()->getLayerLoc()->nf), "Test failed.\n");

   int noPost = getPost()->getLayerLoc()->nf;
   noPost     = (int)params->value(name, "noPost", noPost);
   FatalIf(!(noPost > 0), "Test failed.\n");
   FatalIf(!(noPost <= getPost()->getLayerLoc()->nf), "Test failed.\n");

   float sigma_cocirc = PI / 2.0f;
   sigma_cocirc       = params->value(name, "sigmaCocirc", sigma_cocirc);

   float sigma_kurve = 1.0f; // fraction of delta_radius_curvature
   sigma_kurve       = params->value(name, "sigmaKurve", sigma_kurve);

   // sigma_chord = % of PI * R, where R == radius of curvature (1/curvature)
   float sigma_chord = 0.5f;
   sigma_chord       = params->value(name, "sigmaChord", sigma_chord);

   float delta_theta_max = PI / 2.0f;
   delta_theta_max       = params->value(name, "deltaThetaMax", delta_theta_max);

   float cocirc_self = (getPre() != getPost());
   cocirc_self       = params->value(name, "cocircSelf", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   float delta_radius_curvature = 1.0f; // 1 = minimum radius of curvature
   delta_radius_curvature = params->value(name, "deltaRadiusCurvature", delta_radius_curvature);

   int patchSizeOverall = mWeightsPair->getPreWeights()->getPatchSizeOverall();
   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      cocircCalcWeights(
            &dataStart[patchIndex * patchSizeOverall],
            patchIndex,
            noPre,
            noPost,
            sigma_cocirc,
            sigma_kurve,
            sigma_chord,
            delta_theta_max,
            cocirc_self,
            delta_radius_curvature,
            numFlanks,
            shift,
            aspect,
            rotate,
            sigma,
            r2Max,
            strength);
   }
}

void SharedConnDebugInitWeights::cocircCalcWeights(
      float *dataStart,
      int dataPatchIndex,
      int noPre,
      int noPost,
      float sigma_cocirc,
      float sigma_kurve,
      float sigma_chord,
      float delta_theta_max,
      float cocirc_self,
      float delta_radius_curvature,
      int numFlanks,
      float shift,
      float aspect,
      float rotate,
      float sigma,
      float r2Max,
      float strength) {

   const float min_weight    = 0.0f; // read in as param
   const float sigma2        = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;

   const int nxPatch = getPatchSizeX();
   const int nyPatch = getPatchSizeY();
   const int nfPatch = getPatchSizeF();
   if (nxPatch * nyPatch * nfPatch == 0) {
      return; // reduced patch size is zero
   }

   // get strides of (potentially shrunken) patch
   const int sx = getPatchStrideX();
   FatalIf(!(sx == nfPatch), "Test failed.\n");
   const int sf = getPatchStrideF();
   FatalIf(!(sf == 1), "Test failed.\n");

   float *w_tmp = dataStart;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = getPatchSizeX();
   const int nyPatch_tmp = getPatchSizeY();
   const int nfPatch_tmp = getPatchSizeF();
   int kxKernelIndex;
   int kyKerneIndex;
   int kfKernelIndex;
   this->dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKerneIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKerneIndex;
   const int sx_tmp    = getPatchStrideX();
   FatalIf(!(sx_tmp == getPatchSizeF()), "Test failed.\n");
   const int sy_tmp = getPatchStrideY();
   FatalIf(!(sy_tmp == getPatchSizeF() * nxPatch_tmp), "Test failed.\n");
   const int sf_tmp = getPatchStrideF();
   FatalIf(!(sf_tmp == 1), "Test failed.\n");

   // get distances to nearest neighbor in post synaptic layer
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(
         kxPre_tmp,
         getPost()->getXScale() - getPre()->getXScale(),
         &xDistNNPreUnits,
         &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(
         kyPre_tmp,
         getPost()->getYScale() - getPre()->getYScale(),
         &yDistNNPreUnits,
         &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre_tmp, getPost()->getXScale() - getPre()->getXScale());
   kyNN = nearby_neighbor(kyPre_tmp, getPost()->getYScale() - getPre()->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, getPost()->getXScale() - getPre()->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, getPost()->getYScale() - getPre()->getYScale());

   // get distance to patch head
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale =
         xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits / xDistNNPostUnits;
   float xDistHeadPreUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale =
         yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits / yDistNNPostUnits;
   float yDistHeadPreUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;

   // sigma is in units of pre-synaptic layer
   const float dxPost = powf(2, getPost()->getXScale());
   const float dyPost = powf(2, getPost()->getYScale());

   const int kfPre = featureIndex(
         dataPatchIndex,
         getPre()->getLayerLoc()->nx,
         getPre()->getLayerLoc()->ny,
         getPre()->getLayerLoc()->nf);

   bool POS_KURVE_FLAG  = false; //  handle pos and neg curvature separately
   bool SADDLE_FLAG     = false; // handle saddle points separately
   const int nKurvePre  = getPre()->getLayerLoc()->nf / noPre;
   const int nKurvePost = getPost()->getLayerLoc()->nf / noPost;
   const float dThPre   = PI / noPre;
   const float dThPost  = PI / noPost;
   const float th0Pre   = rotate * dThPre / 2.0f;
   const float th0Post  = rotate * dThPost / 2.0f;
   const int iThPre     = dataPatchIndex % noPre;
   // const int iThPre = kfPre / nKurvePre;
   const float thetaPre = th0Pre + iThPre * dThPre;

   int iKvPre        = kfPre % nKurvePre;
   bool iPosKurvePre = false;
   bool iSaddlePre   = false;
   float radKurvPre  = delta_radius_curvature + iKvPre * delta_radius_curvature;
   float kurvePre    = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   int iKvPreAdj     = iKvPre;
   if (POS_KURVE_FLAG) {
      FatalIf(!(nKurvePre >= 2), "Test failed.\n");
      iPosKurvePre = iKvPre >= (int)(nKurvePre / 2);
      if (SADDLE_FLAG) {
         FatalIf(!(nKurvePre >= 4), "Test failed.\n");
         iSaddlePre = (iKvPre % 2 == 0) ? 0 : 1;
         iKvPreAdj  = ((iKvPre % (nKurvePre / 2)) / 2);
      }
      else { // SADDLE_FLAG
         iKvPreAdj = (iKvPre % (nKurvePre / 2));
      }
   } // POS_KURVE_FLAG
   radKurvPre             = delta_radius_curvature + iKvPreAdj * delta_radius_curvature;
   kurvePre               = (radKurvPre != 0.0f) ? 1 / radKurvPre : 1.0f;
   float sigma_kurve_pre  = sigma_kurve * radKurvPre;
   float sigma_kurve_pre2 = 2 * sigma_kurve_pre * sigma_kurve_pre;
   sigma_chord *= PI * radKurvPre;
   float sigma_chord2 = 2.0f * sigma_chord * sigma_chord;

   // loop over all post synaptic neurons in patch
   for (int kfPost = 0; kfPost < nfPatch_tmp; kfPost++) {
      // int iThPost = kfPost / nKurvePost;
      int iThPost     = kfPost % noPost;
      float thetaPost = th0Post + iThPost * dThPost;

      int iKvPost        = kfPost % nKurvePost;
      bool iPosKurvePost = false;
      bool iSaddlePost   = false;
      float radKurvPost  = delta_radius_curvature + iKvPost * delta_radius_curvature;
      float kurvePost    = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      int iKvPostAdj     = iKvPost;
      if (POS_KURVE_FLAG) {
         FatalIf(!(nKurvePost >= 2), "Test failed.\n");
         iPosKurvePost = iKvPost >= (int)(nKurvePost / 2);
         if (SADDLE_FLAG) {
            FatalIf(!(nKurvePost >= 4), "Test failed.\n");
            iSaddlePost = (iKvPost % 2 == 0) ? 0 : 1;
            iKvPostAdj  = ((iKvPost % (nKurvePost / 2)) / 2);
         }
         else { // SADDLE_FLAG
            iKvPostAdj = (iKvPost % (nKurvePost / 2));
         }
      } // POS_KURVE_FLAG
      radKurvPost             = delta_radius_curvature + iKvPostAdj * delta_radius_curvature;
      kurvePost               = (radKurvPost != 0.0f) ? 1 / radKurvPost : 1.0f;
      float sigma_kurve_post  = sigma_kurve * radKurvPost;
      float sigma_kurve_post2 = 2 * sigma_kurve_post * sigma_kurve_post;

      float deltaTheta = fabsf(thetaPre - thetaPost);
      deltaTheta       = (deltaTheta <= PI / 2.0f) ? deltaTheta : PI - deltaTheta;
      if (deltaTheta > delta_theta_max) {
         continue;
      }

      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = (yDistHeadPreUnits + jPost * dyPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = (xDistHeadPreUnits + iPost * dxPost);

            float gDist      = 0.0f;
            float gChord     = 1.0f;
            float gCocirc    = 1.0f;
            float gKurvePre  = 1.0f;
            float gKurvePost = 1.0f;

            // rotate the reference frame by th
            float dxP = +xDelta * cosf(thetaPre) + yDelta * sinf(thetaPre);
            float dyP = -xDelta * sinf(thetaPre) + yDelta * cosf(thetaPre);

            // include shift to flanks
            float dyP_shift  = dyP - shift;
            float dyP_shift2 = dyP + shift;
            float d2         = dxP * dxP + aspect * dyP * aspect * dyP;
            float d2_shift   = dxP * dxP + (aspect * (dyP_shift)*aspect * (dyP_shift));
            float d2_shift2  = dxP * dxP + (aspect * (dyP_shift2)*aspect * (dyP_shift2));
            if (d2_shift <= r2Max) {
               gDist += expf(-d2_shift / sigma2);
            }
            if (numFlanks > 1) {
               // include shift in opposite direction
               if (d2_shift2 <= r2Max) {
                  gDist += expf(-d2_shift2 / sigma2);
               }
            }
            if (gDist == 0.0f)
               continue;
            if (d2 == 0) {
               bool sameLoc = (kfPre == kfPost);
               if ((!sameLoc) || (cocirc_self)) {
                  gCocirc = sigma_cocirc > 0
                                  ? expf(-deltaTheta * deltaTheta / sigma_cocirc2)
                                  : expf(-deltaTheta * deltaTheta / sigma_cocirc2) - 1.0f;
                  if ((nKurvePre > 1) && (nKurvePost > 1)) {
                     gKurvePre =
                           expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost) / 2
                                * (sigma_kurve_pre * sigma_kurve_pre
                                   + sigma_kurve_post * sigma_kurve_post));
                  }
               }
               else { // sameLoc && !cocircSelf
                  gCocirc = 0.0f;
                  continue;
               }
            }
            else { // d2 > 0

               float atanx2_shift =
                     thetaPre + 2.0f * atan2f(dyP_shift, dxP); // preferred angle (rad)
               atanx2_shift += 2.0f * PI;
               atanx2_shift    = fmodf(atanx2_shift, PI);
               atanx2_shift    = fabsf(atanx2_shift - thetaPost);
               float chi_shift = atanx2_shift; // fabsf(atanx2_shift - thetaPost); // radians
               if (chi_shift >= PI / 2.0f) {
                  chi_shift = PI - chi_shift;
               }
               if (noPre > 1 && noPost > 1) {
                  gCocirc = sigma_cocirc2 > 0 ? expf(-chi_shift * chi_shift / sigma_cocirc2)
                                              : expf(-chi_shift * chi_shift / sigma_cocirc2) - 1.0f;
               }

               // compute curvature of cocircular contour
               float cocircKurve_shift = d2_shift > 0 ? fabsf(2 * dyP_shift) / d2_shift : 0.0f;
               if (POS_KURVE_FLAG) {
                  if (SADDLE_FLAG) {
                     if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift > 0)) {
                        continue;
                     }
                     if ((iPosKurvePre) && (iSaddlePre)
                         && (((dyP_shift > 0) && (dxP < 0)) || ((dyP_shift > 0) && (dxP < 0)))) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (iSaddlePre)
                         && (((dyP_shift > 0) && (dxP > 0)) || ((dyP_shift < 0) && (dxP < 0)))) {
                        continue;
                     }
                  }
                  else { // SADDLE_FLAG
                     if ((iPosKurvePre) && (dyP_shift < 0)) {
                        continue;
                     }
                     if (!(iPosKurvePre) && (dyP_shift > 0)) {
                        continue;
                     }
                  }
               } // POS_KURVE_FLAG
               gKurvePre = (nKurvePre > 1) ? expf(-powf((cocircKurve_shift - fabsf(kurvePre)), 2)
                                                  / sigma_kurve_pre2)
                                           : 1.0f;
               gKurvePost = ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0))
                                  ? expf(-powf((cocircKurve_shift - fabsf(kurvePost)), 2)
                                         / sigma_kurve_post2)
                                  : 1.0f;

               // compute distance along contour
               float d_chord_shift = (cocircKurve_shift != 0.0f) ? atanx2_shift / cocircKurve_shift
                                                                 : sqrtf(d2_shift);
               gChord = (nKurvePre > 1) ? expf(-powf(d_chord_shift, 2) / sigma_chord2) : 1.0f;

               if (numFlanks > 1) {
                  float atanx2_shift2 =
                        thetaPre + 2.0f * atan2f(dyP_shift2, dxP); // preferred angle (rad)
                  atanx2_shift2 += 2.0f * PI;
                  atanx2_shift2    = fmodf(atanx2_shift2, PI);
                  atanx2_shift2    = fabsf(atanx2_shift2 - thetaPost);
                  float chi_shift2 = atanx2_shift2; // fabsf(atanx2_shift2 - thetaPost); // radians
                  if (chi_shift2 >= PI / 2.0f) {
                     chi_shift2 = PI - chi_shift2;
                  }
                  if (noPre > 1 && noPost > 1) {
                     gCocirc += sigma_cocirc2 > 0
                                      ? expf(-chi_shift2 * chi_shift2 / sigma_cocirc2)
                                      : expf(-chi_shift2 * chi_shift2 / sigma_cocirc2) - 1.0f;
                  }

                  float cocircKurve_shift2 =
                        d2_shift2 > 0 ? fabsf(2 * dyP_shift2) / d2_shift2 : 0.0f;
                  if (POS_KURVE_FLAG) {
                     if (SADDLE_FLAG) {
                        if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                        if ((iPosKurvePre) && (iSaddlePre)
                            && (((dyP_shift2 > 0) && (dxP < 0))
                                || ((dyP_shift2 > 0) && (dxP < 0)))) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (iSaddlePre)
                            && (((dyP_shift2 > 0) && (dxP > 0))
                                || ((dyP_shift2 < 0) && (dxP < 0)))) {
                           continue;
                        }
                     }
                     else { // SADDLE_FLAG
                        if ((iPosKurvePre) && (dyP_shift2 < 0)) {
                           continue;
                        }
                        if (!(iPosKurvePre) && (dyP_shift2 > 0)) {
                           continue;
                        }
                     } // SADDLE_FLAG
                  } // POS_KURVE_FLAG
                  gKurvePre += (nKurvePre > 1)
                                     ? expf(-powf((cocircKurve_shift2 - fabsf(kurvePre)), 2)
                                            / sigma_kurve_pre2)
                                     : 1.0f;
                  gKurvePost += ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0))
                                      ? expf(-powf((cocircKurve_shift2 - fabsf(kurvePost)), 2)
                                             / sigma_kurve_post2)
                                      : 1.0f;

                  float d_chord_shift2 = cocircKurve_shift2 != 0.0f
                                               ? atanx2_shift2 / cocircKurve_shift2
                                               : sqrtf(d2_shift2);
                  gChord += (nKurvePre > 1) ? expf(-powf(d_chord_shift2, 2) / sigma_chord2) : 1.0f;
               }
            }
            float weight_tmp = gDist * gKurvePre * gKurvePost * gCocirc;
            if (weight_tmp < min_weight)
               continue;
            w_tmp[iPost * sx_tmp + jPost * sy_tmp + kfPost * sf_tmp] = weight_tmp;
         }
      }
   }
}

void SharedConnDebugInitWeights::initializeGaussian2DWeights(float *dataStart, int numPatches) {
   PVParams *params = parent->parameters();

   // default values (chosen for center on cell of one pixel)
   int noPost          = getPatchSizeF();
   float aspect        = 1.0f; // circular (not line oriented)
   float sigma         = 0.8f;
   float rMax          = 1.4f;
   float rMin          = 0.0f;
   float strength      = 1.0f;
   float deltaThetaMax = 2.0f * PI; // max difference in orientation between pre and post
   float thetaMax      = 1.0f; // max orientation in units of PI
   int numFlanks       = 1;
   float shift         = 0.0f;
   float rotate        = 0.0f; // rotate so that axis isn't aligned
   float bowtieFlag    = 0.0f; // flag for setting bowtie angle
   float bowtieAngle   = PI * 2.0f; // bowtie angle

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   rMin     = params->value(name, "rMin", rMin);
   strength = params->value(name, "strength", strength);
   if (getPatchSizeF() > 1) {
      noPost        = (int)params->value(getName(), "numOrientationsPost", getPatchSizeF());
      deltaThetaMax = params->value(name, "deltaThetaMax", deltaThetaMax);
      thetaMax      = params->value(name, "thetaMax", thetaMax);
      numFlanks     = (int)params->value(name, "numFlanks", (float)numFlanks);
      shift         = params->value(name, "flankShift", shift);
      rotate        = params->value(name, "rotate", rotate);
      bowtieFlag    = params->value(name, "bowtieFlag", bowtieFlag);
      if (bowtieFlag == 1.0f) {
         bowtieAngle = params->value(name, "bowtieAngle", bowtieAngle);
      }
   }

   float r2Max = rMax * rMax;
   float r2Min = rMin * rMin;

   int patchSizeOverall = mWeightsPair->getPreWeights()->getPatchSizeOverall();
   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      gauss2DCalcWeights(
            &dataStart[patchIndex * patchSizeOverall],
            patchIndex,
            noPost,
            numFlanks,
            shift,
            rotate,
            aspect,
            sigma,
            r2Max,
            r2Min,
            strength,
            deltaThetaMax,
            thetaMax,
            bowtieFlag,
            bowtieAngle);
   }
}

void SharedConnDebugInitWeights::gauss2DCalcWeights(
      float *dataStart,
      int dataPatchIndex,
      int no,
      int numFlanks,
      float shift,
      float rotate,
      float aspect,
      float sigma,
      float r2Max,
      float r2Min,
      float strength,
      float deltaThetaMax,
      float thetaMax,
      float bowtieFlag,
      float bowtieAngle) {

   bool self = (getPre() != getPost());

   // get dimensions of (potentially shrunken patch)
   const int nxPatch = getPatchSizeX();
   const int nyPatch = getPatchSizeY();
   const int nfPatch = getPatchSizeF();
   if (nxPatch * nyPatch * nfPatch == 0) {
      return; // reduced patch size is zero
   }

   // float * w = wp->data;

   // get strides of (potentially shrunken) patch
   const int sx = getPatchStrideX();
   FatalIf(!(sx == nfPatch), "Test failed.\n");
   // const int sy = getPatchStrideY(); // no assert here because patch may be shrunken
   const int sf = getPatchStrideF();
   FatalIf(!(sf == 1), "Test failed.\n");

   // make full sized temporary patch, positioned around center of unit cell
   // Patch * wp_tmp;
   // wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   // float * w_tmp = wp_tmp->data;
   float *w_tmp = dataStart; // wp_tmp->data;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = getPatchSizeX();
   const int nyPatch_tmp = getPatchSizeY();
   const int nfPatch_tmp = getPatchSizeF();
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   this->dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKernelIndex;
   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp    = getPatchStrideX();
   FatalIf(!(sx_tmp == getPatchSizeF()), "Test failed.\n");
   const int sy_tmp = getPatchStrideY();
   FatalIf(!(sy_tmp == getPatchSizeF() * nxPatch_tmp), "Test failed.\n");
   const int sf_tmp = getPatchStrideF();
   FatalIf(!(sf_tmp == 1), "Test failed.\n");

   // get distances to nearest neighbor in post synaptic layer (measured relative to pre-synaptic
   // cell)
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(
         kxPre_tmp,
         getPost()->getXScale() - getPre()->getXScale(),
         &xDistNNPreUnits,
         &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(
         kyPre_tmp,
         getPost()->getYScale() - getPre()->getYScale(),
         &yDistNNPreUnits,
         &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor(kxPre_tmp, getPost()->getXScale() - getPre()->getXScale());
   kyNN = nearby_neighbor(kyPre_tmp, getPost()->getYScale() - getPre()->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, getPost()->getXScale() - getPre()->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, getPost()->getYScale() - getPre()->getYScale());

   // get distance to patch head (measured relative to pre-synaptic cell)
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale =
         xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits / xDistNNPostUnits;
   float xDistHeadPreUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale =
         yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits / yDistNNPostUnits;
   float yDistHeadPreUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;

   // sigma is in units of pre-synaptic layer
   const float dxPost = xRelativeScale;
   const float dyPost = yRelativeScale;

   // TODO - the following assumes that if aspect > 1, # orientations = # features
   //   int noPost = no;
   // number of orientations only used if aspect != 1
   const int noPost    = getPost()->getLayerLoc()->nf;
   const float dthPost = PI * thetaMax / (float)noPost;
   const float th0Post = rotate * dthPost / 2.0f;
   const int noPre     = getPre()->getLayerLoc()->nf;
   const float dthPre  = PI * thetaMax / (float)noPre;
   const float th0Pre  = rotate * dthPre / 2.0f;
   const int fPre      = dataPatchIndex % getPre()->getLayerLoc()->nf;
   FatalIf(!(fPre == kfPre_tmp), "Test failed.\n");
   const int iThPre  = dataPatchIndex % noPre;
   const float thPre = th0Pre + iThPre * dthPre;

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      int oPost    = fPost % noPost;
      float thPost = th0Post + oPost * dthPost;
      if (noPost == 1 && noPre > 1) {
         thPost = thPre;
      }
      // TODO: add additional weight factor for difference between thPre and thPost
      if (fabsf(thPre - thPost) > deltaThetaMax) {
         continue;
      }
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = (yDistHeadPreUnits + jPost * dyPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = (xDistHeadPreUnits + iPost * dxPost);
            bool sameLoc = ((fPre == fPost) && (xDelta == 0.0f) && (yDelta == 0.0f));
            if ((sameLoc) && (!self)) {
               continue;
            }

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);
            //            float xp = xDelta * cosf(thPost) - yDelta * sinf(thPost);
            //            float yp = xDelta * sinf(thPost) + yDelta * cosf(thPost);

            if (bowtieFlag == 1.0f) {
               float offaxis_angle = atan2(yp, xp);
               if (((offaxis_angle > bowtieAngle) && (offaxis_angle < (PI - bowtieAngle)))
                   || ((offaxis_angle < -bowtieAngle) && (offaxis_angle > (-PI + bowtieAngle)))) {
                  continue;
               }
            }

            // include shift to flanks
            float d2     = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index    = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if ((d2 <= r2Max) && (d2 >= r2Min)) {
               w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if ((d2 <= r2Max) && (d2 >= r2Min)) {
                  w_tmp[index] += expf(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }
}

void SharedConnDebugInitWeights::initializeGaborWeights(float *dataStart, int numPatches) {

   const int xScale = getPost()->getXScale() - getPre()->getXScale();
   const int yScale = getPost()->getYScale() - getPre()->getYScale();

   PVParams *params = parent->parameters();

   float aspect   = 4.0f;
   float sigma    = 2.0f;
   float rMax     = 8.0f;
   float lambda   = sigma / 0.8f; // gabor wavelength
   float strength = 1.0f;
   float phi      = 0;

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   lambda   = params->value(name, "lambda", lambda);
   strength = params->value(name, "strength", strength);
   phi      = params->value(name, "phi", phi);

   float r2Max = rMax * rMax;

   int patchSizeOverall = mWeightsPair->getPreWeights()->getPatchSizeOverall();
   for (int kernelIndex = 0; kernelIndex < numPatches; kernelIndex++) {
      // TODO - change parameters based on kernelIndex (i.e., change orientation)
      gaborWeights(
            &dataStart[kernelIndex * patchSizeOverall],
            xScale,
            yScale,
            aspect,
            sigma,
            r2Max,
            lambda,
            strength,
            phi);
   }
}

void SharedConnDebugInitWeights::gaborWeights(
      float *dataStart,
      int xScale,
      int yScale,
      float aspect,
      float sigma,
      float r2Max,
      float lambda,
      float strength,
      float phi) {
   PVParams *params = parent->parameters();

   float rotate = 1.0f;
   float invert = 0.0f;
   if (params->present(name, "rotate"))
      rotate = params->value(name, "rotate");
   if (params->present(name, "invert"))
      invert = params->value(name, "invert");

   float *w = dataStart; // wp->data;

   // const float phi = 3.1416;  // phase

   const int nx = getPatchSizeX(); //(int) wp->nx;
   const int ny = getPatchSizeY(); //(int) wp->ny;
   const int nf = getPatchSizeF();

   const int sx = getPatchStrideX(); // FatalIf(!(sx == nf), "Test failed.\n");
   const int sy = getPatchStrideY(); // FatalIf(!(sy == nf*nx), "Test failed.\n");
   const int sf = getPatchStrideF(); // FatalIf(!(sf == 1), "Test failed.\n");

   const float dx = powf(2, xScale);
   const float dy = powf(2, yScale);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   const float x0 = -(nx / 2.0f - 0.5f) * dx;
   // const float y0 = +(ny/2.0 - 0.5) * dy;
   const float y0 = -(ny / 2.0f - 0.5f) * dy;

   const float dth = PI / nf;
   const float th0 = rotate * dth / 2.0f;

   for (int f = 0; f < nf; f++) {
      float th = th0 + f * dth;
      for (int j = 0; j < ny; j++) {
         // float yp = y0 - j * dy;    // pixel coordinate
         float yp = y0 + j * dy; // pixel coordinate
         for (int i = 0; i < nx; i++) {
            float xp = x0 + i * dx; // pixel coordinate

            // rotate the reference frame by th ((x,y) is center of patch (0,0))
            // float u1 = - (0.0f - xp) * cos(th) - (0.0f - yp) * sin(th);
            // float u2 = + (0.0f - xp) * sin(th) - (0.0f - yp) * cos(th);
            float u1 = +xp * cosf(th) + yp * sinf(th);
            float u2 = -xp * sinf(th) + yp * cosf(th);

            float factor = cos(2.0f * PI * u2 / lambda + phi);
            if (fabsf(u2 / lambda) > 3.0f / 4.0f)
               factor = 0.0f; // phase < 3*PI/2 (no second positive band)
            float d2  = u1 * u1 + (aspect * u2 * aspect * u2);
            float wt  = factor * expf(-d2 / (2.0f * sigma * sigma));

#ifdef DEBUG_OUTPUT
            if (j == 0)
               InfoLog().printf("x=%f fac=%f w=%f\n", xp, factor, wt);
#endif
            if (xp * xp + yp * yp > r2Max) {
               w[i * sx + j * sy + f * sf] = 0.0f;
            }
            else {
               if (invert)
                  wt *= -1.0f;
               if (wt < 0.0f)
                  wt                       = 0.0f; // clip negative values
               w[i * sx + j * sy + f * sf] = wt;
            }
         }
      }
   }
}

int SharedConnDebugInitWeights::dataIndexToUnitCellIndex(int dataIndex, int *kx, int *ky, int *kf) {
   Weights *weights          = mWeightsPair->getPreWeights();
   PVLayerLoc const &preLoc  = weights->getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = weights->getGeometry()->getPostLoc();

   int xDataIndex, yDataIndex, fDataIndex;
   if (weights->getSharedFlag()) {

      int nxData = weights->getNumDataPatchesX();
      int nyData = weights->getNumDataPatchesY();
      int nfData = weights->getNumDataPatchesF();
      pvAssert(nfData == preLoc.nf);

      xDataIndex = kxPos(dataIndex, nxData, nyData, nfData);
      yDataIndex = kyPos(dataIndex, nxData, nyData, nfData);
      fDataIndex = featureIndex(dataIndex, nxData, nyData, nfData);
   }
   else { // nonshared weights.
      // data index is extended presynaptic index; convert to restricted.
      int nxExt  = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
      int nyExt  = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
      xDataIndex = kxPos(dataIndex, nxExt, nyExt, preLoc.nf) - preLoc.halo.lt;
      yDataIndex = kyPos(dataIndex, nxExt, nyExt, preLoc.nf) - preLoc.halo.up;
      fDataIndex = featureIndex(dataIndex, nxExt, nyExt, preLoc.nf);
   }
   int xStride = (preLoc.nx > postLoc.nx) ? preLoc.nx / postLoc.nx : 1;
   pvAssert(xStride > 0);

   int yStride = (preLoc.ny > postLoc.ny) ? preLoc.ny / postLoc.ny : 1;
   pvAssert(yStride > 0);

   int xUnitCell = xDataIndex % xStride;
   if (xUnitCell < 0) {
      xUnitCell += xStride;
   }
   pvAssert(xUnitCell >= 0 and xUnitCell < xStride);

   int yUnitCell = yDataIndex % yStride;
   if (yUnitCell < 0) {
      yUnitCell += yStride;
   }
   pvAssert(yUnitCell >= 0 and yUnitCell < yStride);

   int kUnitCell = kIndex(xUnitCell, yUnitCell, fDataIndex, xStride, yStride, preLoc.nf);

   if (kx) {
      *kx = xUnitCell;
   }
   if (ky) {
      *ky = yUnitCell;
   }
   if (kf) {
      *kf = fDataIndex;
   }
   return kUnitCell;
}

} /* namespace PV */
