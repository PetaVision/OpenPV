/*
 * HyPerConnDebugInitWeights.cpp
 *
 *  Created on: Aug 16, 2011
 *      Author: kpeterson
 */

#include "HyPerConnDebugInitWeights.hpp"
#include <normalizers/NormalizeBase.hpp>


namespace PV {

HyPerConnDebugInitWeights::HyPerConnDebugInitWeights()
{
   initialize_base();
}

HyPerConnDebugInitWeights::HyPerConnDebugInitWeights(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) : HyPerConn()
{
   initialize_base();
   HyPerConnDebugInitWeights::initialize(name, hc, weightInitializer, weightNormalizer);
}

HyPerConnDebugInitWeights::~HyPerConnDebugInitWeights()
{
   free(otherConnName);
}

int HyPerConnDebugInitWeights::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer);
   return PV_SUCCESS;
}

int HyPerConnDebugInitWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_copiedConn(ioFlag);
   return status;
}

void HyPerConnDebugInitWeights::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      channel = CHANNEL_INH;
      parent->parameters()->handleUnnecessaryParameter(name, "channelCode", (int) channel);
   }
}

void HyPerConnDebugInitWeights::ioParam_copiedConn(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "copiedConn", &otherConnName);
}

int HyPerConnDebugInitWeights::initialize_base() {
   otherConnName = NULL;
   otherConn=NULL;
   return PV_SUCCESS;
}

int HyPerConnDebugInitWeights::communicateInitInfo() {
   HyPerConn::communicateInitInfo();
   BaseConnection * baseConn = parent->getConnFromName(otherConnName);
   otherConn = dynamic_cast<HyPerConn *>(baseConn);
   if (otherConn == NULL) {
      fprintf(stderr, "HyPerConnDebugInitWeights \"%s\" error in rank %d process: copiedConn \"%s\" is not a connection in the column.\n",
            name, parent->columnId(), otherConnName);
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

PVPatch *** HyPerConnDebugInitWeights::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches, const char * filename)
{
   // TODO  Implement InitWeightsMethod class.  The constructor for HyPerConn would take an InitWeightsMethod
   //       instantiation as an argument.  The routines called below would be put into derived classes
   //       of InitWeightsMethod.
   PVParams * inputParams = parent->parameters();
   PVPatch ** patches = arbors[0];
   pvdata_t * arborStart = dataStart[0];
   numPatches=getNumDataPatches();
   //PVPatch ** kpatches = kernelPatches;
   //int arbor = 0;
   //int numKernelPatches = numDataPatches(arbor);

   int initFromLastFlag = inputParams->value(getName(), "initFromLastFlag", 0.0f, false) != 0;

   if (initFromLastFlag) {
      fprintf(stderr, "This method is for testing weight initialization!  It does not support load from file!\n");
   }
   else {
      const char * weightInitTypeStr = inputParams->stringValue(name, "weightInitType");
      if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "CoCircWeight"))) {
         initializeCocircWeights(patches, arborStart, numPatches);
      }
      else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "SmartWeight"))) {
         initializeSmartWeights(patches, arborStart, numPatches);
      }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "UniformRandomWeight"))) {
      //	      weightInitializer = new InitUniformRandomWeights();
      //	   }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "GaussianRandomWeight"))) {
      //	      weightInitializer = new InitGaussianRandomWeights();
      //	   }
      else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "GaborWeight"))) {
         initializeGaborWeights(patches, arborStart, numPatches);
      }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "PoolWeight"))) {
      //	      weightInitializer = new InitPoolWeights();
      //	   }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "RuleWeight"))) {
      //	      weightInitializer = new InitRuleWeights();
      //	   }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "SubUnitWeight"))) {
      //	      weightInitializer = new InitSubUnitWeights();
      //	   }
      //	   else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "InitIdentWeight"))) {
      //	      weightInitializer = new InitIdentWeights();
      //	   }
      else if(( weightInitTypeStr!=0 )&&(!strcmp(weightInitTypeStr, "Gauss2DWeight"))) {
         initializeGaussian2DWeights(patches, arborStart, numPatches);
      }
      else { //default is also Gauss2D
         //fprintf(stderr, "weightInitType not set or unrecognized.  Using default (2D Gaussian).\n");
         initializeGaussian2DWeights(patches, arborStart, numPatches);
      }

   }

   if (normalizer) {
      normalizer->normalizeWeightsWrapper();
   }
   return arbors;
}

PVPatch ** HyPerConnDebugInitWeights::initializeSmartWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches)
{

   for (int k = 0; k < numPatches; k++) {
      smartWeights(patches[k], dataStart + k*nxp*nyp*nfp, dataIndexToUnitCellIndex(k)); // MA
   }
   return patches;
}
int HyPerConnDebugInitWeights::smartWeights(PVPatch * wp, pvdata_t * dataStart, int k)
{
   pvdata_t * w = dataStart; // wp->data;

   const int nxp = (int) wp->nx;
   const int nyp = (int) wp->ny;
   const int nfp = fPatchSize();

   const int sxp = xPatchStride();
   const int syp = yPatchStride();
   const int sfp = fPatchStride();

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            w[x * sxp + y * syp + f * sfp] = k;
         }
      }
   }

   return 0;
}

PVPatch ** HyPerConnDebugInitWeights::initializeCocircWeights(PVPatch ** patches, pvdata_t * dataStart, int numDataPatches)
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

   int noPre = pre->getLayerLoc()->nf;
   noPre = (int) params->value(name, "noPre", noPre);
   assert(noPre > 0);
   assert(noPre <= pre->getLayerLoc()->nf);

   int noPost = post->getLayerLoc()->nf;
   noPost = (int) params->value(name, "noPost", noPost);
   assert(noPost > 0);
   assert(noPost <= post->getLayerLoc()->nf);

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

   for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
      pvdata_t * patchDataStart = &dataStart[patchIndex*nxp*nyp*nfp];
      cocircCalcWeights(patches[patchIndex], patchDataStart, patchIndex, noPre, noPost, sigma_cocirc,
            sigma_kurve, sigma_chord, delta_theta_max, cocirc_self,
            delta_radius_curvature, numFlanks, shift, aspect, rotate, sigma, r2Max,
            strength);
   }

   return patches;
}
int HyPerConnDebugInitWeights::cocircCalcWeights(PVPatch * wp, pvdata_t * dataStart, int dataPatchIndex, int noPre, int noPost,
      float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
      float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
      float aspect, float rotate, float sigma, float r2Max, float strength)
{
   // pvdata_t * w = wp->data;

   const float min_weight = 0.0f; // read in as param
   const float sigma2 = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;

   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = fPatchSize();
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   // get strides of (potentially shrunken) patch
   const int sx = xPatchStride();
   assert(sx == nfPatch);
   // const int sy = yPatchStride(); // no assert here because patch may be shrunken
   const int sf = fPatchStride();
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   // PVPatch * wp_tmp;
   // wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   // pvdata_t * w_tmp = wp_tmp->data;
   pvdata_t * w_tmp = dataStart;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = nxp; // wp_tmp->nx;
   const int nyPatch_tmp = nyp; // wp_tmp->ny;
   const int nfPatch_tmp = fPatchSize();  // should nfPatch_tmp just be replaced with nfPatch throughout?
   int kxKernelIndex;
   int kyKerneIndex;
   int kfKernelIndex;
   this->dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKerneIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKerneIndex;
   //   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp = xPatchStride();
   assert(sx_tmp == fPatchSize());
   const int sy_tmp = yPatchStride();
   assert(sy_tmp == fPatchSize() * nxPatch_tmp);
   const int sf_tmp = fPatchStride();
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
   const float dxPost = powf(2, post->getXScale());
   const float dyPost = powf(2, post->getYScale());

   //const int kfPre = kPre % pre->clayer->loc.nf;
   const int kfPre = featureIndex(dataPatchIndex, pre->getLayerLoc()->nx, pre->getLayerLoc()->ny,
         pre->getLayerLoc()->nf);

   bool POS_KURVE_FLAG = false; //  handle pos and neg curvature separately
   bool SADDLE_FLAG  = false; // handle saddle points separately
   const int nKurvePre = pre->getLayerLoc()->nf / noPre;
   const int nKurvePost = post->getLayerLoc()->nf / noPost;
   const float dThPre = PI / noPre;
   const float dThPost = PI / noPost;
   const float th0Pre = rotate * dThPre / 2.0;
   const float th0Post = rotate * dThPost / 2.0;
   const int iThPre = dataPatchIndex % noPre;
   //const int iThPre = kfPre / nKurvePre;
   const float thetaPre = th0Pre + iThPre * dThPre;

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
      //int iThPost = kfPost / nKurvePost;
      int iThPost = kfPost % noPost;
      float thetaPost = th0Post + iThPost * dThPost;

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
   // copyToWeightPatch(wp_tmp, 0, kPre);
/*
   w = wp->data;
   const int nxunshrunkPatch = wp_tmp->nx;
   const int nyunshrunkPatch = wp_tmp->ny;
   const int nfunshrunkPatch = fPatchSize();
   const int unshrunkPatchSize = nxunshrunkPatch*nyunshrunkPatch*nfunshrunkPatch;
   pvdata_t *wtop = this->getPatchDataStart(0);
   //pvdata_t * data_head = &wtop[unshrunkPatchSize*kPre];
   //pvdata_t * data_head = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   //size_t data_offset = w - data_head;
   pvdata_t * data_head1 = &wtop[unshrunkPatchSize*kPre]; // (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   pvdata_t * data_head2 = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset1 = w - data_head1;
   size_t data_offset2 = w - data_head2;
   size_t data_offset = fabs(data_offset1) < fabs(data_offset2) ? data_offset1 : data_offset2;
   w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }
*/

   // free(wp_tmp);
   return 0;

}

PVPatch ** HyPerConnDebugInitWeights::initializeGaussian2DWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches)
{
   PVParams * params = parent->parameters();

   // default values (chosen for center on cell of one pixel)
   int noPost = nfp;
   float aspect = 1.0; // circular (not line oriented)
   float sigma = 0.8;
   float rMax = 1.4;
   float rMin = 1.4;
   float strength = 1.0;
   float deltaThetaMax = 2.0f * PI;  // max difference in orientation between pre and post
   float thetaMax = 1.0;  // max orientation in units of PI
   int numFlanks = 1;
   float shift = 0.0f;
   float rotate = 0.0f;   // rotate so that axis isn't aligned
   float bowtieFlag = 0.0f;  // flag for setting bowtie angle
   float bowtieAngle = PI * 2.0f;  // bowtie angle

   aspect   = params->value(name, "aspect", aspect);
   sigma    = params->value(name, "sigma", sigma);
   rMax     = params->value(name, "rMax", rMax);
   rMin     = params->value(name, "rMin", rMin);
   strength = params->value(name, "strength", strength);
   if (nfp > 1) {
      noPost = (int) params->value(getName(), "numOrientationsPost", nfp);
      deltaThetaMax = params->value(name, "deltaThetaMax", deltaThetaMax);
      thetaMax = params->value(name, "thetaMax", thetaMax);
      numFlanks = (int) params->value(name, "numFlanks", (float) numFlanks);
      shift = params->value(name, "flankShift", shift);
      rotate = params->value(name, "rotate", rotate);
      bowtieFlag = params->value(name, "bowtieFlag", bowtieFlag);
      if (bowtieFlag == 1.0f) {
         bowtieAngle = params->value(name, "bowtieAngle", bowtieAngle);
      }
   }

   float r2Max = rMax * rMax;
   float r2Min = rMin * rMin;

   for (int patchIndex = 0; patchIndex < numPatches; patchIndex++) {
      gauss2DCalcWeights(patches[patchIndex], &dataStart[patchIndex*nxp*nyp*nfp], patchIndex, noPost, numFlanks, shift, rotate,
            aspect, sigma, r2Max, r2Min, strength, deltaThetaMax, thetaMax, bowtieFlag, bowtieAngle);
   }

   return patches;
}
int HyPerConnDebugInitWeights::gauss2DCalcWeights(PVPatch * wp, pvdata_t * dataStart, int dataPatchIndex, int no, int numFlanks,
      float shift, float rotate, float aspect, float sigma, float r2Max, float r2Min, float strength,
      float deltaThetaMax, float thetaMax, float bowtieFlag, float bowtieAngle)
{
   //   const PVLayer * lPre = pre->clayer;
   //   const PVLayer * lPost = post->clayer;

   bool self = (pre != post);

   // get dimensions of (potentially shrunken patch)
   const int nxPatch = (int) wp->nx;
   const int nyPatch = (int) wp->ny;
   const int nfPatch = fPatchSize();
   if (nxPatch * nyPatch * nfPatch == 0) {
      return 0; // reduced patch size is zero
   }

   // pvdata_t * w = wp->data;

   // get strides of (potentially shrunken) patch
   const int sx = xPatchStride();
   assert(sx == nfPatch);
   // const int sy = yPatchStride(); // no assert here because patch may be shrunken
   const int sf = fPatchStride();
   assert(sf == 1);

   // make full sized temporary patch, positioned around center of unit cell
   // PVPatch * wp_tmp;
   // wp_tmp = pvpatch_inplace_new(nxp, nyp, nfp);
   pvdata_t * w_tmp = dataStart; // wp_tmp->data;

   // get/check dimensions and strides of full sized temporary patch
   const int nxPatch_tmp = nxp; // wp_tmp->nx;
   const int nyPatch_tmp = nyp; // wp_tmp->ny;
   const int nfPatch_tmp = fPatchSize();
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   this->dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);

   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKernelIndex;
   const int kfPre_tmp = kfKernelIndex;
   const int sx_tmp = xPatchStride();
   assert(sx_tmp == fPatchSize());
   const int sy_tmp = yPatchStride();
   assert(sy_tmp == fPatchSize() * nxPatch_tmp);
   const int sf_tmp = fPatchStride();
   assert(sf_tmp == 1);

   // get distances to nearest neighbor in post synaptic layer (measured relative to pre-synaptic cell)
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre_tmp, pre->getXScale(), post->getXScale(),
         &xDistNNPreUnits, &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kyPre_tmp, pre->getYScale(), post->getYScale(),
         &yDistNNPreUnits, &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor( kxPre_tmp, pre->getXScale(), post->getXScale());
   kyNN = nearby_neighbor( kyPre_tmp, pre->getYScale(), post->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, pre->getYScale(), post->getYScale());

   // get distance to patch head (measured relative to pre-synaptic cell)
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
   const float dxPost = xRelativeScale; //powf(2, (float) post->getXScale());
   const float dyPost = yRelativeScale; //powf(2, (float) post->getYScale());


   // TODO - the following assumes that if aspect > 1, # orientations = # features
   //   int noPost = no;
   // number of orientations only used if aspect != 1
   const int noPost = post->getLayerLoc()->nf;
   const float dthPost = PI*thetaMax / (float) noPost;
   const float th0Post = rotate * dthPost / 2.0f;
   const int noPre = pre->getLayerLoc()->nf;
   const float dthPre = PI*thetaMax / (float) noPre;
   const float th0Pre = rotate * dthPre / 2.0f;
   const int fPre = dataPatchIndex % pre->getLayerLoc()->nf;
   assert(fPre == kfPre_tmp);
   const int iThPre = dataPatchIndex % noPre;
   const float thPre = th0Pre + iThPre * dthPre;

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      int oPost = fPost % noPost;
      float thPost = th0Post + oPost * dthPost;
      if (noPost == 1 && noPre > 1) {
         thPost = thPre;
      }
      //TODO: add additional weight factor for difference between thPre and thPost
      if (fabs(thPre - thPost) > deltaThetaMax) {
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

            if (bowtieFlag == 1.0f){
               float offaxis_angle = atan2(yp, xp);
               if ( ((offaxis_angle > bowtieAngle) && (offaxis_angle < (PI - bowtieAngle))) ||
                     ((offaxis_angle < -bowtieAngle) && (offaxis_angle > (-PI + bowtieAngle))) ){
                  continue;
               }
            }

            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if ((d2 <= r2Max) && (d2 >= r2Min)) {
               w_tmp[index] += expf(-d2
                     / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if ((d2 <= r2Max) && (d2 >= r2Min)) {
                  w_tmp[index] += expf(-d2
                        / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   // copy weights from full sized temporary patch to (possibly shrunken) patch
   // copyToWeightPatch(wp_tmp, 0, kPre);
/*
   w = wp->data;
   const int nxunshrunkPatch = wp_tmp->nx;
   const int nyunshrunkPatch = wp_tmp->ny;
   const int nfunshrunkPatch = fPatchSize();
   const int unshrunkPatchSize = nxunshrunkPatch*nyunshrunkPatch*nfunshrunkPatch;
   pvdata_t *wtop = this->getPatchDataStart(0);
   //pvdata_t * data_head = &wtop[unshrunkPatchSize*kPre];
   //pvdata_t * data_head =  (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   //size_t data_offset = w - data_head;
   pvdata_t * data_head1 = &wtop[unshrunkPatchSize*kPre]; // (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   pvdata_t * data_head2 = (pvdata_t *) ((char*) wp + sizeof(PVPatch));
   size_t data_offset1 = w - data_head1;
   size_t data_offset2 = w - data_head2;
   size_t data_offset = fabs(data_offset1) < fabs(data_offset2) ? data_offset1 : data_offset2;
   w_tmp = &wp_tmp->data[data_offset];
   int nk = nxPatch * nfPatch;
   for (int ky = 0; ky < nyPatch; ky++) {
      for (int iWeight = 0; iWeight < nk; iWeight++) {
         w[iWeight] = w_tmp[iWeight];
      }
      w += sy;
      w_tmp += sy_tmp;
   }
*/

   // free(wp_tmp);
   return 0;
}

PVPatch ** HyPerConnDebugInitWeights::initializeGaborWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches)
{

   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();

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
      gaborWeights(patches[kernelIndex], &dataStart[kernelIndex*nxp*nyp*nfp], xScale, yScale, aspect, sigma, r2Max, lambda, strength, phi);
   }
   return patches;
}

int HyPerConnDebugInitWeights::gaborWeights(PVPatch * wp, pvdata_t * dataStart, int xScale, int yScale,
      float aspect, float sigma, float r2Max, float lambda, float strength, float phi)
{
   PVParams * params = parent->parameters();

   float rotate = 1.0;
   float invert = 0.0;
   if (params->present(name, "rotate")) rotate = params->value(name, "rotate");
   if (params->present(name, "invert")) invert = params->value(name, "invert");

   pvdata_t * w = dataStart; // wp->data;

   //const float phi = 3.1416;  // phase

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = fPatchSize();

   const int sx = xPatchStride();  //assert(sx == nf);
   const int sy = yPatchStride();  //assert(sy == nf*nx);
   const int sf = fPatchStride();  //assert(sf == 1);

   const float dx = powf(2, xScale);
   const float dy = powf(2, yScale);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   const float x0 = -(nx/2.0f - 0.5f) * dx;
   //const float y0 = +(ny/2.0 - 0.5) * dy;
   const float y0 = -(ny/2.0f - 0.5f) * dy;

   const float dth = PI/nf;
   const float th0 = rotate*dth/2.0f;


   for (int f = 0; f < nf; f++) {
      float th = th0 + f * dth;
      for (int j = 0; j < ny; j++) {
         //float yp = y0 - j * dy;    // pixel coordinate
         float yp = y0 + j * dy;    // pixel coordinate
         for (int i = 0; i < nx; i++) {
            float xp  = x0 + i*dx;  // pixel coordinate

            // rotate the reference frame by th ((x,y) is center of patch (0,0))
            //float u1 = - (0.0f - xp) * cos(th) - (0.0f - yp) * sin(th);
            //float u2 = + (0.0f - xp) * sin(th) - (0.0f - yp) * cos(th);
            float u1 = +xp * cosf(th) + yp * sinf(th);
            float u2 = -xp * sinf(th) + yp * cosf(th);

            float factor = cos(2.0f*PI*u2/lambda + phi);
            if (fabs(u2/lambda) > 3.0f/4.0f) factor = 0.0f;  // phase < 3*PI/2 (no second positive band)
            float d2 = u1 * u1 + (aspect*u2 * aspect*u2);
            float wt = factor * expf(-d2 / (2.0f*sigma*sigma));

#ifdef DEBUG_OUTPUT
            if (j == 0) printf("x=%f fac=%f w=%f\n", xp, factor, wt);
#endif
            if (xp*xp + yp*yp > r2Max) {
               w[i*sx + j*sy + f*sf] = 0.0f;
            }
            else {
               if (invert) wt *= -1.0f;
               if (wt < 0.0f) wt = 0.0f;       // clip negative values
               w[i*sx + j*sy + f*sf] = wt;
            }
         }
      }
   }


   return 0;
}

BaseObject * createHyPerConnDebugInitWeights(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = getWeightInitializer(name, hc);
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new HyPerConnDebugInitWeights(name, hc, weightInitializer, weightNormalizer);
}

} /* namespace PV */
