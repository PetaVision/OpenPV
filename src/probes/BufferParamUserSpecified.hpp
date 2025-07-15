#ifndef BUFFERPARAMUSERSPECIFIED_HPP_
#define BUFFERPARAMUSERSPECIFIED_HPP_

#include "probes/BufferParamInterface.hpp"

namespace PV {

class BufferParamUserSpecified : public BufferParamInterface {
  public:
   BufferParamUserSpecified(std::shared_ptr<ParamsIO> paramsIO);

   virtual void ioParam_buffer(ParamsIOSwitch ioSwitch);

  protected:
   void initialize(std::shared_ptr<ParamsIO> paramsIO);
};

} // namespace PV

#endif // BUFFERPARAMUSERSPECIFIED_HPP_
