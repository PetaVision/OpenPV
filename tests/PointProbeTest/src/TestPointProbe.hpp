/*
 * TestPointProbe.hpp
 *
 */

#ifndef TESTPOINTPROBE_HPP_
#define TESTPOINTPROBE_HPP_

#include "probes/PointProbe.hpp"

namespace PV {

class TestPointProbe : public PV::PointProbe {
  public:
   TestPointProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~TestPointProbe();

  protected:
   TestPointProbe();
   virtual int point_writeState(double timef, float outVVal, float outAVal);
}; // end class TestPointProbe

} // end namespace PV

#endif /* POINTPROBE_HPP_ */
