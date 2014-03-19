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
}

int InitWindowed3DGaussWeightsParams::initialize_base() {

   windowShift = 0.0f;
   windowShiftT = -10;


   return 1;
}
int InitWindowed3DGaussWeightsParams::initialize(HyPerConn * parentConn) {
   return Init3DGaussWeightsParams::initialize(parentConn);
}

int InitWindowed3DGaussWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = Init3DGaussWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_windowShiftT(ioFlag);
   ioParam_windowShift(ioFlag);
   return status;
}

void InitWindowed3DGaussWeightsParams::ioParam_windowShiftT(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "windowShiftT", &windowShiftT);
}

void InitWindowed3DGaussWeightsParams::ioParam_windowShift(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "windowShift", &windowShift);
}

} /* namespace PV */
