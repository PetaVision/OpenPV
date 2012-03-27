/*
 * ConnFunctionProbe.cpp
 *
 *  Created on: Mar 23, 2012
 *      Author: pschultz
 */

#include "ConnFunctionProbe.hpp"

namespace PV {

ConnFunctionProbe::ConnFunctionProbe()
{
   initialize_base();
}

ConnFunctionProbe::ConnFunctionProbe(const char * probename, const char * filename, HyPerConn * conn) {
   initialize_base();
   initialize(probename, filename, conn);
}

int ConnFunctionProbe::initialize_base() {
   return PV_SUCCESS;
}

int ConnFunctionProbe::initialize(const char * probename, const char * filename, HyPerConn * conn) {
   BaseConnectionProbe::initialize(probename, filename, conn);
   return PV_SUCCESS;
}

int ConnFunctionProbe::outputState(float timef) {
   int status = PV_SUCCESS;
   if( status == PV_SUCCESS ) {
      double energy = evaluate(timef);
      fprintf(getFilePtr(), "Time %f: Energy %f\n", timef, energy);
   }
   return status;
}

ConnFunctionProbe::~ConnFunctionProbe()
{
}

}  // end namespace PV
