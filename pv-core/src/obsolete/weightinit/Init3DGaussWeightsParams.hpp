/*
 * Init3DGaussWeightsParams.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef INIT3DGAUSSWEIGHTSPARAMS_HPP_
#define INIT3DGAUSSWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

class Init3DGaussWeightsParams: public PV::InitGauss2DWeightsParams {
public:
   Init3DGaussWeightsParams();
   Init3DGaussWeightsParams(HyPerConn * parentConn);
   virtual ~Init3DGaussWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void calcOtherParams(int patchIndex);
   bool isSameLocOrSelf(float xDelta, float yDelta, int fPost);
   bool checkBowtieAngle(float xp, float yp);

   //get-set methods:
   inline float getTAspect()        {return taspect;}
   inline float getYAspect()        {return yaspect;}
   inline float getShiftT()        {return shiftT;}
   inline float getThetaXT()        {return thetaXT;}
   inline int getTime()        {return time;}
   inline void setTime(int t)        {time=(int)(t*dT);}

protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);
   virtual void ioParam_yaspect(enum ParamsIOFlag ioFlag);
   virtual void ioParam_taspect(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dT(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shiftT(enum ParamsIOFlag ioFlag);
   virtual void ioParam_flowSpeed(enum ParamsIOFlag ioFlag);

private:

   //params file values:
   float yaspect; // aspect ratio with respect to y -axis
   float taspect; // aspect ratio with respect to time-axis
   float shiftT;
   bool bowtieFlag;  // flag for setting bowtie angle
   float bowtieAngle;  // bowtie angle
   double dT; //change in delay between arbors

   //calculated values;
   float thetaXT;  //=>angle corresponding to feature velocity
   int time;
};

} /* namespace PV */
#endif /* INIT3DGAUSSWEIGHTSPARAMS_HPP_ */
