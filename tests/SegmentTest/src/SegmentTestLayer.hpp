#ifndef SEGMENTTESTLAYER_HPP_
#define SEGMENTTESTLAYER_HPP_

#include <components/SegmentBuffer.hpp>
#include <layers/SegmentLayer.hpp>

namespace PV {

class SegmentTestLayer : public PV::SegmentLayer {
  public:
   SegmentTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   Response::Status checkUpdateState(double timef, double dt) override;

  private:
   SegmentBuffer *mSegmentBuffer = nullptr;
};

} /* namespace PV */
#endif
