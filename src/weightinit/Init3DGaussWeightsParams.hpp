/*
 * Init3DGaussWeightsParams.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef INIT3DGAUSSWEIGHTSPARAMS_HPP_
#define INIT3DGAUSSWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class Init3DGaussWeightsParams: public PV::InitWeightsParams {
public:
   Init3DGaussWeightsParams();
   Init3DGaussWeightsParams(HyPerConn * parentConn);
   virtual ~Init3DGaussWeightsParams();
   void calcOtherParams(int patchIndex);
   bool isSameLocOrSelf(float xDelta, float yDelta, int fPost);
   bool checkBowtieAngle(float xp, float yp);

   //get-set methods:
   inline float getTAspect()        {return taspect;}
   inline float getYAspect()        {return yaspect;}
   inline float getShift()        {return shift;}
   inline float getShiftT()        {return shiftT;}
   inline int getNumFlanks()        {return numFlanks;}
   inline float getSigma()        {return sigma;}
   inline double getR2Max()        {return r2Max;}
   inline float getThetaXT()        {return thetaXT;}
   inline int getTime()        {return time;}
   inline void setTime(int t)        {time=(int)(t*dT);}
   inline float getStrength()        {return strength;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:

   //params file values:
   float yaspect; // aspect ratio with respect to y -axis
   float taspect; // aspect ratio with respect to time-axis
   float sigma;
   float rMax;
   float strength;
   int numFlanks;
   float shift;
   float shiftT;
   bool bowtieFlag;  // flag for setting bowtie angle
   float bowtieAngle;  // bowtie angle
   double dT; //change in delay between arbors

   //calculated values;
   float thetaXT;  //=>angle corresponding to feature velocity
   double r2Max;
   bool self;
   int time;
};

} /* namespace PV */
#endif /* INIT3DGAUSSWEIGHTSPARAMS_HPP_ */
