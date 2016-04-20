/*
 * TestPointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "TestPointProbe.hpp"
#include <string.h>
#include <columns/HyPerCol.hpp>

namespace PV {

TestPointProbe::TestPointProbe() {
   // Default constructor for derived classes.  Derived classes should call initTestPointProbe from their init-method.
}

TestPointProbe::TestPointProbe(const char * probeName, HyPerCol * hc) :
   PointProbe()
{
   initialize(probeName, hc);
}

TestPointProbe::~TestPointProbe()
{
}


int TestPointProbe::point_writeState(double timef, float outVVal, float outAVal) {
   if(parent->columnId()==0){
      //Input pvp layer's spinning order is nf, nx, ny
      float expectedVal = fLoc * 64 + xLoc * 8 + yLoc;
      if(outAVal != expectedVal){
         std::cout << "Connection " << name << " Mismatch: actual value: " << outAVal << " Expected value: " << expectedVal << ".\n";
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(-1);
      }
   }
   return PV_SUCCESS;
}

BaseObject * createTestPointProbe(char const * name, HyPerCol * hc) {
   return hc ? new TestPointProbe(name, hc) : NULL;
}

} // namespace PV
