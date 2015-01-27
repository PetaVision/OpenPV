/*
 * TestPointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "TestPointProbe.hpp"
#include <string.h>

namespace PV {

TestPointProbe::TestPointProbe() {
   // Default constructor for derived classes.  Derived classes should call initTestPointProbe from their init-method.
}

/**
 * @probeName
 * @hc
 */
TestPointProbe::TestPointProbe(const char * probeName, HyPerCol * hc) :
   PointProbe()
{
}

TestPointProbe::~TestPointProbe()
{
}


/**
 * @time
 * @l
 * @k
 * @kex
 */
int TestPointProbe::point_writeState(double timef, float outVVal, float outAVal) {
   if(parent->columnId()==0){
      //Input pvp layer's spinning order is nf, nx, ny
      float expectedVal = fLoc * 64 + xLoc * 8 + yLoc;
      if(outAVal != expectedvalue){
         std::cout << "Connection " << name << " Mismatch at (" << xval << "," << yval << ") : actual value: " << actualvalue << " Expected value: " << expectedvalue << ".\n";
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(-1);
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
