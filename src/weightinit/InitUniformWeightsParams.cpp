/*
 * InitUniformWeightsParams.cpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#include "InitUniformWeightsParams.hpp"

namespace PV {

InitUniformWeightsParams::InitUniformWeightsParams()
{
   initialize_base();
}

InitUniformWeightsParams::InitUniformWeightsParams(HyPerConn * parentConn)
                     : InitWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitUniformWeightsParams::~InitUniformWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitUniformWeightsParams::initialize_base() {

   initWeight = 0;
   connectOnlySameFeatures = false;
   return 1;
}

int InitUniformWeightsParams::initialize(HyPerConn * parentConn) {
   InitWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   initWeight = params->value(getName(), "weightInit", initWeight);
   connectOnlySameFeatures = params->value(getName(), "connectOnlySameFeatures", connectOnlySameFeatures);


   return status;

}

} /* namespace PV */
