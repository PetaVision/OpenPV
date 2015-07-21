/*
 * InitOneToOneWeightsWithDelaysParams.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: wchavez
 */

#include "InitOneToOneWeightsWithDelaysParams.hpp"

namespace PV {

InitOneToOneWeightsWithDelaysParams::InitOneToOneWeightsWithDelaysParams()
{
   initialize_base();
}
InitOneToOneWeightsWithDelaysParams::InitOneToOneWeightsWithDelaysParams(const char * name, HyPerCol * hc)
                     : InitWeightsParams() {
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeightsWithDelaysParams::~InitOneToOneWeightsWithDelaysParams()
{
}

int InitOneToOneWeightsWithDelaysParams::initialize_base() {

   initWeight = 1;
   return 1;
}

int InitOneToOneWeightsWithDelaysParams::initialize(const char * name, HyPerCol * hc) {
   return InitWeightsParams::initialize(name, hc);
}

int InitOneToOneWeightsWithDelaysParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InitWeightsParams::ioParamsFillGroup(ioFlag);
   ioParam_weightInit(ioFlag);
   return status;
}

void InitOneToOneWeightsWithDelaysParams::ioParam_weightInit(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, getName(), "weightInit", &initWeight, initWeight);
}

void InitOneToOneWeightsWithDelaysParams::calcOtherParams(int patchIndex) {

   this->getcheckdimensionsandstrides();

   //const int kfPre_tmp = this->kernalIndexCalculations(patch, patchIndex);



   //this->calculateThetas(kfPre_tmp, patchIndex);

}


} /* namespace PV */
