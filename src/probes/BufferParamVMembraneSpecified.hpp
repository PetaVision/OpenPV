#ifndef BUFFERPARAMVMEMBRANESPECIFIED_HPP_
#define BUFFERPARAMVMEMBRANESPECIFIED_HPP_

#include "BufferParamInterface.hpp"

namespace PV {

class BufferParamVMembraneSpecified : public BufferParamInterface {
  public:
   BufferParamVMembraneSpecified(std::shared_ptr<ParamsIO> paramsIO);

   virtual ~BufferParamVMembraneSpecified();

   virtual void ioParam_buffer(ParamsIOSwitch ioSwitch) override;

  protected:
   void initialize(std::shared_ptr<ParamsIO> paramsIO);
};

} // namespace PV

#endif // BUFFERPARAMVMEMBRANESPECIFIED_HPP_
