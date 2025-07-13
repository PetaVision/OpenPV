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

class InitCocircWeights : public InitGauss2DWeights {
  protected:
   virtual void ioParam_sigmaCocirc(ParamsIOSwitch ioSwitch);
   virtual void ioParam_sigmaKurve(ParamsIOSwitch ioSwitch);
   virtual void ioParam_cocircSelf(ParamsIOSwitch ioSwitch);
   virtual void ioParam_deltaRadiusCurvature(ParamsIOSwitch ioSwitch);

  public:
   InitCocircWeights(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~InitCocircWeights();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual void calcWeights(int patchIndex, int arborId) override;

  protected:
   InitCocircWeights();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  private:
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
   void cocircCalcWeights(float *dataStart);

  private:
   float mSigmaCocirc          = 0.5f * PI;
   float mSigmaKurve           = 1.0f; // fraction of delta_radius_curvature
   float mCocircSelf           = false;
   float mDeltaRadiusCurvature = 1.0f; // 1 = minimum radius of curvature
   float mMinWeight            = 0.0f; // read in as param
   bool mPosKurveFlag          = false; //  handle pos and neg curvature separately
   bool mSaddleFlag            = false; // handle saddle points separately

   // calculated parameters:
   int mNKurvePre;
   bool mIPosKurvePre;
   bool mISaddlePre;
   float mKurvePre;
   int mNKurvePost;
   bool mIPosKurvePost;
   bool mISaddlePost;
   float mKurvePost;
   // float mSigmaKurvePre; // Unused data member
   float mSigmaKurvePre2;
   float mSigmaKurvePost;
   float mSigmaKurvePost2;

   // used for calculating weights:
   float mGDist;
   float mGCocirc;
   float mGKurvePre;
   float mGKurvePost;

}; // class InitCocircWeights

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTS_HPP_ */
