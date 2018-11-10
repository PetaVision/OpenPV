#ifndef SEGMENTIFYTEST_HPP_
#define SEGMENTIFYTEST_HPP_

#include <components/SegmentifyBuffer.hpp>
#include <layers/Segmentify.hpp>

namespace PV {

class SegmentifyTest : public PV::Segmentify {
  public:
   SegmentifyTest(const char *name, HyPerCol *hc);

  protected:
   virtual void createComponentTable(char const *description) override;
   Response::Status updateState(double timef, double dt) override;

  private:
   float getTargetVal(int yi, int xi, int fi);
   int checkOutputVals(int yi, int xi, int fi, float targetVal, float actualVal);

  private:
   SegmentifyBuffer *mSegmentifyBuffer = nullptr;
};

} /* namespace PV */
#endif
