/*
 * Arguments.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#include <cstdlib>
#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif
#include "Arguments.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <istream>

namespace PV {

Arguments::Arguments(std::istream &configStream, bool allowUnrecognizedArguments) {
   initialize_base();
   initialize(configStream, allowUnrecognizedArguments);
}

int Arguments::initialize_base() { return PV_SUCCESS; }

int Arguments::initialize(std::istream &configStream, bool allowUnrecognizedArguments) {
   resetState(configStream, allowUnrecognizedArguments);
   return PV_SUCCESS;
}

void Arguments::resetState(std::istream &configStream, bool allowUnrecognizedArguments) {
   delete mConfigFromStream;
   mConfigFromStream = new ConfigParser(configStream, allowUnrecognizedArguments);
   resetState();
}

void Arguments::resetState() { mCurrentConfig = mConfigFromStream->getConfig(); }

int Arguments::printState() const {
   InfoLog() << mCurrentConfig.printConfig();
   return PV_SUCCESS;
}

Arguments::~Arguments() { delete mConfigFromStream; }

} /* namespace PV */
