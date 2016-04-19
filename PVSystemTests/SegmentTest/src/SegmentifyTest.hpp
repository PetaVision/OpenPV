#ifndef SEGMENTIFYTEST_HPP_ 
#define SEGMENTIFYTEST_HPP_

#include <layers/Segmentify.hpp>

namespace PV {

class SegmentifyTest: public PV::Segmentify{
public:
   SegmentifyTest(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
   float getTargetVal(int yi, int xi, int fi);
   int checkOutputVals(int yi, int xi, int fi, float targetVal, float actualVal);
};

BaseObject * createSegmentifyTest(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
