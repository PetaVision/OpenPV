/*
 * AverageRateConn.cpp
 *
 *  Created on: Aug 24, 2012
 *      Author: pschultz
 */

#include "AverageRateConn.hpp"

namespace PV {

AverageRateConn::AverageRateConn() {
   initialize_base();
}

AverageRateConn::AverageRateConn(const char * name, HyPerCol * hc,  const char * pre_layer_name, const char * post_layer_name) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name);
}

AverageRateConn::~AverageRateConn() {
}

int AverageRateConn::initialize_base() {
   return PV_SUCCESS;
}

int AverageRateConn::initialize(const char * name, HyPerCol * hc,  const char * pre_layer_name, const char * post_layer_name) {
   int status = IdentConn::initialize(name, hc, pre_layer_name, post_layer_name, NULL);
   return status;
}

int AverageRateConn::setParams(PVParams * inputParams) {
   int status = IdentConn::setParams(inputParams);
   plasticityFlag = true;
   return status;
}

int AverageRateConn::updateState(double timed, double dt) {
   float t = timed <= dt ? dt : timed; // Avoid dividing by zero.
   float w = 1/t;
   int arbor = 0; // Assumes one axonal arbor.
   assert(nfp==getNumDataPatches());
   assert(nxp==1 && nyp==1);
   for (int k = 0; k < getNumDataPatches(); k++) {
      pvdata_t * p = get_wDataHead(arbor, k);
      p[k] = w;
   }
   return PV_SUCCESS;
}

} /* namespace PV */
