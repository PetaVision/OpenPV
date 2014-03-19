/*
 * NormalizeGap.cpp
 *
 *  Created on: Feb 28, 2014
 *      Author: pschultz
 */

#include "NormalizeGap.hpp"

namespace PV {



NormalizeGap::NormalizeGap() {
   initialize_base();
}

NormalizeGap::NormalizeGap(GapConn * callingConn) {
   initialize_base();
   initialize(callingConn);
}

NormalizeGap::~NormalizeGap() {
}

int NormalizeGap::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeGap::initialize(GapConn * callingConn) {
   int status = NormalizeSum::initialize(callingConn);
   return status;
}

void NormalizeGap::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizeFromPostPerspective = true;
      parent()->parameters()->handleUnnecessaryParameter(name, "normalizeFromPostPerspective", true);
   }
}

} /* namespace PV */
