#ifndef BUFFERPARAMACTIVITYSPECIFIED_HPP_
#define BUFFERPARAMACTIVITYSPECIFIED_HPP_

#include "BufferParamInterface.hpp"
#include "io/PVParams.hpp"

namespace PV {

class BufferParamActivitySpecified : public BufferParamInterface {
  public:
   BufferParamActivitySpecified(char const *name, PVParams *params);
   virtual ~BufferParamActivitySpecified();

   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  protected:
   void initialize(char const *name, PVParams *params);
};

} // namespace PV

#endif // BUFFERPARAMACTIVITYSPECIFIED_HPP_
