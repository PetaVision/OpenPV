/*
 * TestPointProbe.hpp
 *
 */

#ifndef TESTPOINTPROBE_HPP_
#define TESTPOINTPROBE_HPP_

#include <io/PointProbe.hpp>

namespace PV {

class TestPointProbe: public PV::PointProbe{
public:
   TestPointProbe(const char * probeName, HyPerCol * hc);
   virtual ~TestPointProbe();

protected:
   TestPointProbe();
   virtual int point_writeState(double timef, float outVVal, float outAVal);
}; // end class TestPointProbe

BaseObject * createTestPointProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* POINTPROBE_HPP_ */
