/*
 * InitGaborWeightsParams.cpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#include "InitGaborWeightsParams.hpp"

namespace PV {

InitGaborWeightsParams::InitGaborWeightsParams()
{
   initialize_base();
}
InitGaborWeightsParams::InitGaborWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitGaborWeightsParams::~InitGaborWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitGaborWeightsParams::initialize_base() {

   aspect = 4.0; // circular (not line oriented)
   sigma = 2.0;
   rMax = 8.0;
   strength = 1.0;
   r2Max = rMax * rMax;

   //numFlanks = 1;
   shift = 0.0f;
   //rotate = 1.0f; // rotate so that axis isn't aligned
   setRotate(0.0f); // rotate so that axis isn't aligned

   lambda = (int)(sigma/0.8);
   phi=0;
   invert=true;

   setThetaMax(1.0f); // max orientation in units of PI

   return 1;
}

int InitGaborWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   aspect = params->value(name, "aspect", aspect);
   sigma = params->value(name, "sigma", sigma);
   rMax = params->value(name, "rMax", rMax);
   lambda   = (int)params->value(name, "lambda", lambda);
   strength = params->value(name, "strength", strength);
   phi = params->value(name, "phi", phi);
   double rMaxd = (double) rMax;
   r2Max = rMaxd * rMaxd;

   shift = params->value(name, "flankShift", shift);
   //rotate = params->value(name, "rotate", rotate);
   setRotate(params->value(name, "rotate", getRotate()));
   invert = (bool)params->value(name, "invert", invert);

   return status;

}

void InitGaborWeightsParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   const int kfPre_tmp = this->kernelIndexCalculations(patchIndex);



   this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
