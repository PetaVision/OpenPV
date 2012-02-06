/*
 * InitCocircWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITCOCIRCWEIGHTSPARAMS_HPP_
#define INITCOCIRCWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitCocircWeightsParams: public PV::InitWeightsParams {
public:
   InitCocircWeightsParams();
   InitCocircWeightsParams(HyPerConn * parentConn);
   virtual ~InitCocircWeightsParams();
   void calcOtherParams(PVPatch * patch, int patchIndex);

   //get/set methods:
   inline float getaspect()        {return aspect;}
   inline float getshift()        {return shift;}
   inline int getnumFlanks()        {return numFlanks;}
   inline double getr2Max()        {return r2Max;}
   inline float getsigma()        {return sigma;}
   inline float getSigma_cocirc()        {return sigma_cocirc;}
   inline float getSigma_kurve()        {return sigma_kurve;}
   inline float getSigma_kurve_pre()        {return sigma_kurve_pre;}
   inline float getSigma_kurve_pre2()        {return sigma_kurve_pre2;}
   inline float getSigma_kurve_post2()        {return sigma_kurve_post2;}
   inline float getSigma_chord()        {return sigma_chord;}
   inline float getmin_weight()        {return min_weight;}
   inline float getnKurvePre()        {return nKurvePre;}
   inline float getGDist()        {return gDist;}

   float calcKurvePreAndSigmaKurvePre();
   float calcKurvePostAndSigmaKurvePost(int kfPost);
   float calcKurveAndSigmaKurve(int kf, int &nKurve,
         float &sigma_kurve_temp, float &kurve_tmp,
         bool &iPosKurve, bool &iSaddle);
   bool checkSameLoc(int kfPost);
   bool checkFlags(float dyP_shift, float dxP);
   void updateCocircNChord(
         float thPost, float dyP_shift, float dxP, float cocircKurve_shift,
         float d2_shift);
   void updategKurvePreNgKurvePost(float cocircKurve_shift);
   void initializeDistChordCocircKurvePreAndKurvePost();
   float calculateWeight();
   void addToGDist(float inc);

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);



private:
   //params variables:
   float aspect; // circular (not line oriented)
   float sigma;
   float rMax;
   double r2Max;
   float strength;
   int numFlanks;
   float shift;
   //float rotate; // rotate so that axis isn't aligned
   //int noPre;
   //int noPost;
   float sigma_cocirc;
   float sigma_kurve; // fraction of delta_radius_curvature
   float sigma_chord;
   //float delta_theta_max;
   float cocirc_self;
   float delta_radius_curvature; // 1 = minimum radius of curvature

   //these variables have hard coded values!  Should the be read in as params?
   float min_weight; // read in as param
   bool POS_KURVE_FLAG; //  handle pos and neg curvature separately
   bool SADDLE_FLAG; // handle saddle points separately

   //calculated parameters:
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

   //used for calculating weights:
   float gDist;
   float gChord; //not used!
   float gCocirc;
   float gKurvePre;
   float gKurvePost;


};

} /* namespace PV */
#endif /* INITCOCIRCWEIGHTSPARAMS_HPP_ */
