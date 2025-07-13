#ifndef BUFFERPARAMUSERSPECIFIED_HPP_
#define BUFFERPARAMUSERSPECIFIED_HPP_

#include "probes/BufferParamInterface.hpp"

namespace PV {

class BufferParamUserSpecified : public BufferParamInterface {
  public:
   BufferParamUserSpecified(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

   virtual void ioParam_buffer(ParamsIOSwitch ioSwitch);

  protected:
   void initialize(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
};

} // namespace PV

#endif // BUFFERPARAMUSERSPECIFIED_HPP_
