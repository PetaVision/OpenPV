/*
 * InitMTWeightsParams.hpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#ifndef INITMTWEIGHTSPARAMS_HPP_
#define INITMTWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "InitGauss2DWeightsParams.hpp"

namespace PV {

class InitMTWeightsParams: public PV::InitGauss2DWeightsParams {
public:
   InitMTWeightsParams();
   InitMTWeightsParams(HyPerConn * parentConn);
   virtual ~InitMTWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateParamsInfo();
   void calcOtherParams(int patchIndex);

   virtual float calcDthPre();
   virtual float calcTh0Pre(float dthPre);

   //get-set methods:
   inline float getSpeed()        {return tunedSpeed;}
   inline float getV1Speed()        {return inputV1Speed;}

protected:
   int initialize_base();
   int initialize(HyPerConn * parentConn);
   void ioParam_nfpRelatedParams(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tunedSpeed(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputV1Speed(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputV1Rotate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_inputV1ThetaMax(enum ParamsIOFlag ioFlag);

private:
   float tunedSpeed;
   float inputV1Speed;
   float inputV1Rotate;
   float inputV1ThetaMax;
};

} /* namespace PV */
#endif /* INITMTWEIGHTSPARAMS_HPP_ */
