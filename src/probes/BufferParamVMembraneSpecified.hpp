#ifndef BUFFERPARAMVMEMBRANESPECIFIED_HPP_
#define BUFFERPARAMVMEMBRANESPECIFIED_HPP_

#include "BufferParamInterface.hpp"
#include "io/PVParams.hpp"

namespace PV {

class BufferParamVMembraneSpecified : public BufferParamInterface {
  public:
   BufferParamVMembraneSpecified(char const *name, PVParams *params);
   virtual ~BufferParamVMembraneSpecified();

   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  protected:
   void initialize(char const *name, PVParams *params);
};

} // namespace PV

#endif // BUFFERPARAMVMEMBRANESPECIFIED_HPP_
