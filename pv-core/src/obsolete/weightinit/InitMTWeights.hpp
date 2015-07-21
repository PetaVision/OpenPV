/*
 * InitMTWeights.hpp
 *
 *  Created on: Oct 25, 2011
 *      Author: kpeterson
 */

#ifndef INITMTWEIGHTS_HPP_
#define INITMTWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitGauss2DWeights.hpp"
#include "InitWeightsParams.hpp"
#include "InitMTWeightsParams.hpp"

namespace PV {

class InitWeightsParams;
class InitMTWeightsParams;

class InitMTWeights: public PV::InitGauss2DWeights {
public:
   InitMTWeights(HyPerConn * conn);
   virtual ~InitMTWeights();
   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();


protected:
   InitMTWeights();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int calculateMTWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitMTWeightsParams * weightParamPtr);
   int calculateVector(float theta, float speed, float &x, float &y, float &t);
   int calculateMTPlane(float theta, float speed, float &x, float &y, float &t);
   int calculate2ndVector(float p1x, float p1y, float p1t, float &p2x, float &p2y, float &p2t);
   float calcDist(float v1x, float v1y, float v1t, float mtx, float mty, float mtt);
};

} /* namespace PV */
#endif /* INITMTWEIGHTS_HPP_ */
