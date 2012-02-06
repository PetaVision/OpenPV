/*
 * InitMTWeights.hpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#ifndef INITMTWEIGHTS_HPP_
#define INITMTWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"
#include "InitMTWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitMTWeightsParams;

class InitMTWeights: public PV::InitWeights {
public:
   InitMTWeights();
   virtual ~InitMTWeights();
   virtual int calcWeights(PVPatch * patch, int patchIndex, int arborId,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);


protected:
   virtual int initialize_base();

private:
   int calculateMTWeights(PVPatch * patch, InitMTWeightsParams * weightParamPtr);
   int calculateVector(float theta, float speed, float &x, float &y, float &t);
   int calculateMTPlane(float theta, float speed, float &x, float &y, float &t);
   int calculate2ndVector(float p1x, float p1y, float p1t, float &p2x, float &p2y, float &p2t);
   float calcDist(float v1x, float v1y, float v1t, float mtx, float mty, float mtt);
};

} /* namespace PV */
#endif /* INITMTWEIGHTS_HPP_ */
