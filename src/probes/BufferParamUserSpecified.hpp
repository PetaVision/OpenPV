#ifndef BUFFERPARAMUSERSPECIFIED_HPP_
#define BUFFERPARAMUSERSPECIFIED_HPP_

#include "io/PVParams.hpp"
#include "probes/BufferParamInterface.hpp"

namespace PV {

class BufferParamUserSpecified : public BufferParamInterface {
  public:
   BufferParamUserSpecified(char const *name, PVParams *params);

   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

  protected:
   void initialize(char const *name, PVParams *params);
};

} // namespace PV

#endif // BUFFERPARAMUSERSPECIFIED_HPP_
