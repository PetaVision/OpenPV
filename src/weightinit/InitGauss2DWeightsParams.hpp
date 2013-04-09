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
public:
   InitGauss2DWeightsParams();
   InitGauss2DWeightsParams(HyPerConn * parentConn);
   virtual ~InitGauss2DWeightsParams();
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
   inline float getaspect()        {return aspect;}
   inline float getshift()        {return shift;}
   inline int getnumFlanks()        {return numFlanks;}
   inline float getsigma()        {return sigma;}
   inline double getr2Max()        {return r2Max;}
   inline double getr2Min()        {return r2Min;}
   inline double getStrength()        {return strength;}
   inline float getthPre()        {return thPre;}
   inline int getFPre()        {return fPre;}

   virtual float calcDthPre();
   virtual float calcTh0Pre(float dthPre);
   float calcThPost(int fPost);
   bool checkTheta(float thPost);


protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);
   void calculateThetas(int kfPre_tmp, int patchIndex);

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


private:

public:
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

   //calculated values;
   double r2Max;
   double r2Min;
   bool self;

};

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTSPARAMS_HPP_ */
