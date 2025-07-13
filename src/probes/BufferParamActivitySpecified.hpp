#ifndef BUFFERPARAMACTIVITYSPECIFIED_HPP_
#define BUFFERPARAMACTIVITYSPECIFIED_HPP_

#include "BufferParamInterface.hpp"

namespace PV {

class BufferParamActivitySpecified : public BufferParamInterface {
  public:
   BufferParamActivitySpecified(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~BufferParamActivitySpecified();

   virtual void ioParam_buffer(ParamsIOSwitch ioSwitch) override;

  protected:
   void initialize(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
};

} // namespace PV

#endif // BUFFERPARAMACTIVITYSPECIFIED_HPP_
