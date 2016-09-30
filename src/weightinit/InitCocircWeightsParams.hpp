/*
 * InitCocircWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTSPARAMS_HPP_
#define INITCOCIRCWEIGHTSPARAMS_HPP_

#include "InitGauss2DWeightsParams.hpp"
#include "InitWeightsParams.hpp"

namespace PV {

class InitCocircWeightsParams : public PV::InitGauss2DWeightsParams {
  public:
   InitCocircWeightsParams();
   InitCocircWeightsParams(const char *name, HyPerCol *hc);
   virtual ~InitCocircWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);

   // get/set methods:
   inline float getSigma_cocirc() { return sigma_cocirc; }
   inline float getSigma_kurve() { return sigma_kurve; }
   inline float getSigma_kurve_pre() { return sigma_kurve_pre; }
   inline float getSigma_kurve_pre2() { return sigma_kurve_pre2; }
   inline float getSigma_kurve_post2() { return sigma_kurve_post2; }
   inline float getmin_weight() { return min_weight; }
   inline float getnKurvePre() { return nKurvePre; }
   inline float getGDist() { return gDist; }

   float calcKurvePostAndSigmaKurvePost(int kfPost);
   float calcKurveAndSigmaKurve(
         int kf,
         int &nKurve,
         float &sigma_kurve_temp,
         float &kurve_tmp,
         bool &iPosKurve,
         bool &iSaddle);
   bool checkSameLoc(int kfPost);
   bool checkFlags(float dyP_shift, float dxP);
   void updateCocircNChord(
         float thPost,
         float dyP_shift,
         float dxP,
         float cocircKurve_shift,
         float d2_shift);
   void updategKurvePreNgKurvePost(float cocircKurve_shift);
   void initializeDistChordCocircKurvePreAndKurvePost();
   float calculateWeight();
   void addToGDist(float inc);

  protected:
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_sigmaCocirc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sigmaKurve(enum ParamsIOFlag ioFlag);
   virtual void ioParam_cocircSelf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaRadiusCurvature(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numOrientationsPre(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numOrientationsPost(enum ParamsIOFlag ioFlag);

  private:
   // params variables:
   float aspect; // circular (not line oriented)
   float sigma;
   float rMax;
   double r2Max;
   float strength;
   int numFlanks;
   float shift;
   float sigma_cocirc;
   float sigma_kurve; // fraction of delta_radius_curvature
   float cocirc_self;
   float delta_radius_curvature; // 1 = minimum radius of curvature

   // these variables have hard coded values!  Should the be read in as params?
   float min_weight; // read in as param
   bool POS_KURVE_FLAG; //  handle pos and neg curvature separately
   bool SADDLE_FLAG; // handle saddle points separately

   // calculated parameters:
   int nKurvePre;
   bool iPosKurvePre;
   bool iSaddlePre;
   float kurvePre;
   int nKurvePost;
   bool iPosKurvePost;
   bool iSaddlePost;
   float kurvePost;
   float sigma_kurve_pre;
   float sigma_kurve_pre2;
   float sigma_kurve_post;
   float sigma_kurve_post2;

   // used for calculating weights:
   float gDist;
   float gCocirc;
   float gKurvePre;
   float gKurvePost;
};

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTSPARAMS_HPP_ */
