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

AverageRateConn::AverageRateConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

AverageRateConn::~AverageRateConn() {
}

int AverageRateConn::initialize_base() {
   return PV_SUCCESS;
}

int AverageRateConn::initialize(const char * name, HyPerCol * hc) {
   int status = IdentConn::initialize(name, hc);
   return status;
}

int AverageRateConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = IdentConn::ioParamsFillGroup(ioFlag);
   return status;
}

void AverageRateConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = true;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

int AverageRateConn::updateState(double timed, double dt) {
   float t = timed <= dt ? dt : timed; // Avoid dividing by zero.
   float w = 1/t;
   int arbor = 0; // Assumes one axonal arbor.
   assert(nfp==getNumDataPatches());
   assert(nxp==1 && nyp==1);
   for (int k = 0; k < getNumDataPatches(); k++) {
      pvwdata_t * p = get_wDataHead(arbor, k);
      //TODO-CER-2014.4.4 - weight conversion
      p[k] = w;
   }
   return PV_SUCCESS;
}

BaseObject * createAverageRateConn(char const * name, HyPerCol * hc) {
   return hc ? new AverageRateConn(name, hc) : NULL;
}

} /* namespace PV */
