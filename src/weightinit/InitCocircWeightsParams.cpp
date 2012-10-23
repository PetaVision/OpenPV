/*
 * InitCocircWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitCocircWeightsParams.hpp"

namespace PV {

InitCocircWeightsParams::InitCocircWeightsParams()
{
   initialize_base();
}
InitCocircWeightsParams::InitCocircWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitCocircWeightsParams::~InitCocircWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitCocircWeightsParams::initialize_base() {

   aspect = 1.0; // circular (not line oriented)
   sigma = 0.8;
   rMax = 1.4;
   strength = 1.0;
   r2Max = rMax * rMax;

   numFlanks = 1;
   shift = 0.0f;
   setRotate(0.0f); // rotate so that axis isn't aligned
   //setDeltaThetaMax(2.0f * PI);  // max orientation in units of PI
   setThetaMax(1.0f); // max orientation in units of PI

   sigma_cocirc = PI / 2.0;

   sigma_kurve = 1.0; // fraction of delta_radius_curvature

   // sigma_chord = % of PI * R, where R == radius of curvature (1/curvature)
   sigma_chord = 0.5;

   setDeltaThetaMax(PI / 2.0);

   cocirc_self = (pre != post);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   delta_radius_curvature = 1.0; // 1 = minimum radius of curvature

   //why are these hard coded in!!!:
   min_weight = 0.0f; // read in as param
   POS_KURVE_FLAG = false; //  handle pos and neg curvature separately
   SADDLE_FLAG  = false; // handle saddle points separately

   return 1;
}

int InitCocircWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   aspect = params->value(name, "aspect", aspect);
   sigma = params->value(name, "sigma", sigma);
   rMax = params->value(name, "rMax", rMax);
   strength = params->value(name, "strength", strength);
   double rMaxd = (double) rMax;
   r2Max = rMaxd * rMaxd;

   numFlanks = (int) params->value(name, "numFlanks", numFlanks);
   shift = params->value(name, "flankShift", shift);
   setRotate(params->value(name, "rotate", getRotate()));

   int noPreTmp = pre->getLayerLoc()->nf;
   noPreTmp = (int) params->value(name, "noPre", noPreTmp);
   assert(noPreTmp > 0);
   assert(noPreTmp <= pre->getLayerLoc()->nf);
   setNoPre(noPreTmp);

//
   int noPostTmp = post->getLayerLoc()->nf;
   noPostTmp = (int) params->value(name, "noPost", noPostTmp);
   assert(noPostTmp > 0);
   assert(noPostTmp <= post->getLayerLoc()->nf);
   setNoPost(noPostTmp);

   sigma_cocirc = params->value(name, "sigmaCocirc", sigma_cocirc);

   sigma_kurve = params->value(name, "sigmaKurve", sigma_kurve);

   // sigma_chord = % of PI * R, where R == radius of curvature (1/curvature)
   sigma_chord = params->value(name, "sigmaChord", sigma_chord);

   float delta_theta_max_tmp = params->value(name, "deltaThetaMax", getDeltaThetaMax());
   setDeltaThetaMax(delta_theta_max_tmp);
   cocirc_self = params->value(name, "cocircSelf", cocirc_self);

   // from pv_common.h
   // // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   delta_radius_curvature = params->value(name, "deltaRadiusCurvature",
         delta_radius_curvature);


   return status;

}

void InitCocircWeightsParams::calcOtherParams(int patchIndex) {
   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);
   nKurvePre = pre->getLayerLoc()->nf / getNoPre();
   nKurvePost = post->getLayerLoc()->nf / getNoPost();
   this->calculateThetas(kfPre_tmp, patchIndex);
   float radKurvPre = this->calcKurvePreAndSigmaKurvePre();

   sigma_chord *= PI * radKurvPre;

}
float InitCocircWeightsParams::calcKurvePreAndSigmaKurvePre() {
   int iKvPre = this->getFPre() % nKurvePre;
   float radKurvPre = calcKurveAndSigmaKurve(iKvPre, nKurvePre,
         sigma_kurve_pre, kurvePre,
         iPosKurvePre, iSaddlePre);
   sigma_kurve_pre2 = 2 * sigma_kurve_pre * sigma_kurve_pre;
   return radKurvPre;
}
float InitCocircWeightsParams::calcKurvePostAndSigmaKurvePost(int kfPost) {
   int iKvPost = kfPost % nKurvePost;
   float radKurvPost = calcKurveAndSigmaKurve(iKvPost, nKurvePost,
         sigma_kurve_post, kurvePost,
         iPosKurvePost, iSaddlePost);
   sigma_kurve_post2 = 2 * sigma_kurve_post * sigma_kurve_post;
   return radKurvPost;
}

float InitCocircWeightsParams::calcKurveAndSigmaKurve(int kf, int &nKurve,
      float &sigma_kurve_temp, float &kurve_tmp,
      bool &iPosKurve, bool &iSaddle) {
   int iKv = kf % nKurve;
   iPosKurve = false;
   iSaddle = false;
   float radKurv = delta_radius_curvature + iKv * delta_radius_curvature;
   sigma_kurve_temp = sigma_kurve * radKurv;

   kurve_tmp = (radKurv != 0.0f) ? 1 / radKurv : 1.0f;

   int iKvPostAdj = iKv;
   if (POS_KURVE_FLAG) {
      assert(nKurve >= 2);
      iPosKurve = iKv >= (int) (nKurve / 2);
      if (SADDLE_FLAG) {
         assert(nKurve >= 4);
         iSaddle = (iKv % 2 == 0) ? 0 : 1;
         iKvPostAdj = ((iKv % (nKurve / 2)) / 2);
      }
      else { // SADDLE_FLAG
         iKvPostAdj = (iKv % (nKurve / 2));
      }
   } // POS_KURVE_FLAG
   radKurv = delta_radius_curvature + iKvPostAdj * delta_radius_curvature;
   kurve_tmp = (radKurv != 0.0f) ? 1 / radKurv : 1.0f;
   return radKurv;
}

bool InitCocircWeightsParams::checkSameLoc(int kfPost) {
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;
   bool sameLoc = (getFPre() == kfPost);
   if ((!sameLoc) || (cocirc_self)) {
      gCocirc = sigma_cocirc > 0 ? expf(-getDeltaTheta() * getDeltaTheta()
            / sigma_cocirc2) : expf(-getDeltaTheta() * getDeltaTheta() / sigma_cocirc2)
            - 1.0;
      if ((nKurvePre > 1) && (nKurvePost > 1)) {
         gKurvePre = expf(-(kurvePre - kurvePost) * (kurvePre - kurvePost)
               / (sigma_kurve_pre2 + sigma_kurve_post2));
      }
   }
   else { // sameLoc && !cocircSelf
      gCocirc = 0.0;
      return true;
   }
   return false;
}

bool InitCocircWeightsParams::checkFlags(float dyP_shift, float dxP) {
   if (POS_KURVE_FLAG) {
      if (SADDLE_FLAG) {
         if ((iPosKurvePre) && !(iSaddlePre) && (dyP_shift < 0)) {
            return true;
         }
         if (!(iPosKurvePre) && !(iSaddlePre) && (dyP_shift > 0)) {
            return true;
         }
         if ((iPosKurvePre) && (iSaddlePre)
               && (((dyP_shift > 0) && (dxP < 0)) || ((dyP_shift > 0) && (dxP
                     < 0)))) {
            return true;
         }
         if (!(iPosKurvePre) && (iSaddlePre) && (((dyP_shift > 0)
               && (dxP > 0)) || ((dyP_shift < 0) && (dxP < 0)))) {
            return true;
         }
      }
      else { //SADDLE_FLAG
         if ((iPosKurvePre) && (dyP_shift < 0)) {
            return true;
         }
         if (!(iPosKurvePre) && (dyP_shift > 0)) {
            return true;
         }
      }
   } // POS_KURVE_FLAG
   return false;
}

void InitCocircWeightsParams::updateCocircNChord(
      float thPost, float dyP_shift, float dxP, float cocircKurve_shift,
      float d2_shift) {

   const float thetaPre = getthPre();
   const int noPre = getNoPre();
   const int noPost = getNoPost();
   const float sigma_cocirc2 = 2 * getSigma_cocirc() * getSigma_cocirc();
   const float sigma_chord2 = 2.0 * getSigma_chord() * getSigma_chord();
   const int nKurvePre = (int)getnKurvePre();

   float atanx2_shift = thetaPre + 2. * atan2f(dyP_shift, dxP); // preferred angle (rad)
   atanx2_shift += 2. * PI;
   atanx2_shift = fmodf(atanx2_shift, PI);
   atanx2_shift = fabsf(atanx2_shift - thPost);
   float chi_shift = atanx2_shift; //fabsf(atanx2_shift - thetaPost); // radians
   if (chi_shift >= PI / 2.0) {
      chi_shift = PI - chi_shift;
   }
   if (noPre > 1 && noPost > 1) {
      gCocirc = sigma_cocirc2 > 0 ? expf(-chi_shift * chi_shift
            / sigma_cocirc2) : expf(-chi_shift * chi_shift / sigma_cocirc2)
            - 1.0;
   }
   // compute distance along contour
   float d_chord_shift = (cocircKurve_shift != 0.0f) ? atanx2_shift
         / cocircKurve_shift : sqrt(d2_shift);
   gChord = (nKurvePre > 1) ? expf(-powf(d_chord_shift, 2) / sigma_chord2)
         : 1.0;

}

void InitCocircWeightsParams::updategKurvePreNgKurvePost(float cocircKurve_shift) {

   const float sigma_cocirc2 = 2 * getSigma_cocirc() * getSigma_cocirc();
   const float sigma_kurve_pre2 = getSigma_kurve_pre2();
   const float sigma_kurve_post2 = getSigma_kurve_post2();

   gKurvePre = (nKurvePre > 1) ? expf(-powf((cocircKurve_shift - fabsf(
         kurvePre)), 2) / sigma_kurve_pre2) : 1.0;
   gKurvePost
         = ((nKurvePre > 1) && (nKurvePost > 1) && (sigma_cocirc2 > 0)) ? expf(
               -powf((cocircKurve_shift - fabsf(kurvePost)), 2)
                     / sigma_kurve_post2)
               : 1.0;
}

void InitCocircWeightsParams::initializeDistChordCocircKurvePreAndKurvePost() {
   gDist = 0.0;
   gChord = 1.0; //not used!
   gCocirc = 1.0;
   gKurvePre = 1.0;
   gKurvePost = 1.0;
}

float InitCocircWeightsParams::calculateWeight() {
   return gDist * gKurvePre * gKurvePost * gCocirc;
}

void InitCocircWeightsParams::addToGDist(float inc) {
   gDist+=inc;
}

} /* namespace PV */
