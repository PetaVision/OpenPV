/*
 * InitCocircWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTS_HPP_
#define INITCOCIRCWEIGHTS_HPP_

#include "InitGauss2DWeights.hpp"

namespace PV {

class InitCocircWeights : public PV::InitGauss2DWeights {
  protected:
   virtual void ioParam_sigmaCocirc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sigmaKurve(enum ParamsIOFlag ioFlag);
   virtual void ioParam_cocircSelf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaRadiusCurvature(enum ParamsIOFlag ioFlag);

  public:
   InitCocircWeights(char const *name, HyPerCol *hc);
   virtual ~InitCocircWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void calcWeights(float *dataStart, int patchIndex, int arborId) override;

  protected:
   InitCocircWeights();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
   float calcKurvePostAndSigmaKurvePost(int kfPost);
   float calcKurveAndSigmaKurve(
         int kf,
         int &nKurve,
         float &sigma_kurve_temp,
         float &kurve_tmp,
         bool &iPosKurve,
         bool &iSaddle);
   void initializeDistChordCocircKurvePreAndKurvePost();
   bool calcDistChordCocircKurvePreNKurvePost(float xDelta, float yDelta, int kfPost, float thPost);
   void addToGDist(float inc);
   bool checkSameLoc(int kfPost);
   void updateCocircNChord(
         float thPost,
         float dyP_shift,
         float dxP,
         float cocircKurve_shift,
         float d2_shift);
   bool checkFlags(float dyP_shift, float dxP);
   void updategKurvePreNgKurvePost(float cocircKurve_shift);
   float calculateWeight();
   void cocircCalcWeights(float *w_tmp);

  private:
   float mSigmaCocirc          = 0.5f * PI;
   float mSigmaKurve           = 1.0f; // fraction of delta_radius_curvature
   float mCocircSelf           = false;
   float mDeltaRadiusCurvature = 1.0f; // 1 = minimum radius of curvature
   float mMinWeight            = 0.0f; // read in as param
   bool mPosKurveFlag          = false; //  handle pos and neg curvature separately
   bool mSaddleFlag            = false; // handle saddle points separately

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

}; // class InitCocircWeights

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTS_HPP_ */
