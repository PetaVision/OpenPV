/*
 * InitWindowed3DGaussWeightsParams.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: kpeterson
 */

#include "InitWindowed3DGaussWeightsParams.hpp"

namespace PV {

InitWindowed3DGaussWeightsParams::InitWindowed3DGaussWeightsParams()
{
   initialize_base();
}
InitWindowed3DGaussWeightsParams::InitWindowed3DGaussWeightsParams(HyPerConn * parentConn)
                     : Init3DGaussWeightsParams() {
   initialize_base();
   initialize(parentConn);
}

InitWindowed3DGaussWeightsParams::~InitWindowed3DGaussWeightsParams()
{
   // TODO Auto-generated destructor stub
}

int InitWindowed3DGaussWeightsParams::initialize_base() {

   windowShift = 0.0f;
   windowShiftT = -10;


   return 1;
}
int InitWindowed3DGaussWeightsParams::initialize(HyPerConn * parentConn) {
   Init3DGaussWeightsParams::initialize(parentConn);

   PVParams * params = parent->parameters();
   int status = PV_SUCCESS;

   windowShiftT = params->value(getName(), "windowShiftT", windowShiftT);
   windowShift = params->value(getName(), "windowShift", windowShift);


   return status;

}

} /* namespace PV */
