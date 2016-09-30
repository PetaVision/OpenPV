/*
 * InitGauss2DWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSS2DWEIGHTSPARAMS_HPP_
#define INITGAUSS2DWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitGauss2DWeightsParams: public PV::InitWeightsParams {
protected:

   /** 
    * List of parameters needed from the InitGauss2DWeightParams class
    * @anchor Gauss2DWeightParams
    * @name InitGauss2DWeight Parameters
    * @{
    */
   virtual void ioParam_aspect(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sigma(enum ParamsIOFlag ioFlag);
   virtual void ioParam_rMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_rMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numOrientationsPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numOrientationsPre(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaThetaMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_thetaMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numFlanks(enum ParamsIOFlag ioFlag);
   virtual void ioParam_flankShift(enum ParamsIOFlag ioFlag);
   virtual void ioParam_rotate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_bowtieFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_bowtieAngle(enum ParamsIOFlag ioFlag);
   void ioParam_aspectRelatedParams(enum ParamsIOFlag ioFlag);
   /** @} */
public:
   InitGauss2DWeightsParams();
   InitGauss2DWeightsParams(const char * name, HyPerCol * hc);
   virtual ~InitGauss2DWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateParamsInfo();
   virtual void calcOtherParams(int patchIndex);
   bool isSameLocOrSelf(float xDelta, float yDelta, int fPost);
   bool checkBowtieAngle(float xp, float yp);

   //get-set methods:
   inline int getNoPost()        {return numOrientationsPost;}
   inline int getNoPre()        {return numOrientationsPre;}
   inline float getThetaMax()        {return thetaMax;}
   inline float getDeltaThetaMax()        {return deltaThetaMax;}
   inline float getDeltaTheta()        {return deltaTheta;}
   inline float getRotate()        {return rotate;}
   inline void setThetaMax(float thetaMaxTmp)        {thetaMax=thetaMaxTmp;}
   inline void setDeltaThetaMax(float thetaMaxTmp)        {deltaThetaMax=thetaMaxTmp;}
   inline void setRotate(float rotateTmp)        {rotate=rotateTmp;}
   inline void setNoPre(int noPreTmp)        {numOrientationsPre=noPreTmp;}
   inline void setNoPost(int noPostTmp)        {numOrientationsPost=noPostTmp;}
   inline float getAspect()        {return aspect;}
   inline float getShift()        {return shift;}
   inline int getNumFlanks()        {return numFlanks;}
   inline float getSigma()        {return sigma;}
   inline double getr2Max()        {return r2Max;}
   inline double getr2Min()        {return r2Min;}
   inline double getStrength()        {return strength;}
   inline float getthPre()        {return thPre;}
   inline int getFPre()        {return fPre;}

   virtual float calcDthPre();
   virtual float calcTh0Pre(float dthPre);
   float calcThPost(int fPost);
   bool checkThetaDiff(float thPost);
   bool checkColorDiff(int fPost);

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);


   bool needAspectParams();
   void calculateThetas(int kfPre_tmp, int patchIndex);


private:

protected:
   int numOrientationsPost;
   float dthPost;
   float th0Post;
   int numOrientationsPre;
   int fPre;
   float thPre;
   float thetaMax;  // max orientation in units of PI
   float rotate;   // rotate so that axis isn't aligned
   float deltaThetaMax;  // max orientation in units of PI
   float deltaTheta;

   //params file values:
   float aspect; // set to 1 for circularly symmetric (not oriented)
   float sigma;
   float rMax;
   float rMin;  // minimum radius for any connection
   float strength;
   int numFlanks;
   float shift;
   bool bowtieFlag;  // flag for setting bowtie angle
   float bowtieAngle;  // bowtie angle

   //calculated values
   double r2Max;
   double r2Min;
   bool self;

};

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTSPARAMS_HPP_ */
