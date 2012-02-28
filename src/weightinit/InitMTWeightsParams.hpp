/*
 * InitMTWeightsParams.hpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#ifndef INITMTWEIGHTSPARAMS_HPP_
#define INITMTWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitMTWeightsParams: public PV::InitWeightsParams {
public:
   InitMTWeightsParams();
   InitMTWeightsParams(HyPerConn * parentConn);
   virtual ~InitMTWeightsParams();
   void calcOtherParams(int patchIndex);

   virtual float calcDthPre();
   virtual float calcTh0Pre(float dthPre);

   //get-set methods:
   inline float getSpeed()        {return tunedSpeed;}
   inline void setSpeed(float speed)        {tunedSpeed=speed;}
   inline float getV1Speed()        {return inputV1Speed;}
   inline void setV1Speed(float speed)        {inputV1Speed=speed;}


protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);

private:
   float tunedSpeed;
   float inputV1Speed;
   float inputV1Rotate;
   float inputV1ThetaMax;
};

} /* namespace PV */
#endif /* INITMTWEIGHTSPARAMS_HPP_ */
