/*
 * TestPointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "TestPointProbe.hpp"
#include <columns/HyPerCol.hpp>
#include <string.h>

namespace PV {

TestPointProbe::TestPointProbe() {
   // Default constructor for derived classes.  Derived classes should call initialize from
   // their init-method.
}

TestPointProbe::TestPointProbe(const char *name, HyPerCol *hc) : PointProbe() {
   initialize(name, hc);
}

TestPointProbe::~TestPointProbe() {}

int TestPointProbe::point_writeState(double timef, float outVVal, float outAVal) {
   if (parent->getCommunicator()->commRank() == 0) {
      // Input pvp layer's spinning order is nf, nx, ny
      float expectedVal = fLoc * 64 + xLoc * 8 + yLoc;
      if (outAVal != expectedVal) {
         ErrorLog() << "Connection " << name << " Mismatch: actual value: " << outAVal
                    << " Expected value: " << expectedVal << ".\n";
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(PV_FAILURE);
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
