#ifndef NORMPROBELOCALTEMPLATE_HPP_
#define NORMPROBELOCALTEMPLATE_HPP_

#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "structures/PVLayerLoc.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/NormProbeLocalInterface.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <cstdlib>
#include <memory>

namespace PV {

template <class C>
class NormProbeLocalTemplate : public NormProbeLocalInterface {
  public:
   NormProbeLocalTemplate(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~NormProbeLocalTemplate() {}

   virtual void initializeState(HyPerLayer *targetLayer) override;

  protected:
   NormProbeLocalTemplate() {}
   virtual std::shared_ptr<C const> createCostFunctionSum() { return nullptr; }
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

  private:
   void calculateNorms(double simTime, ProbeData<double> &values) const override;

  private:
   std::shared_ptr<C const> mCostFunctionSum = nullptr;
};

template <class C>
NormProbeLocalTemplate<C>::NormProbeLocalTemplate(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

template <class C>
void NormProbeLocalTemplate<C>::calculateNorms(double simTime, ProbeData<double> &values) const {
   C const *norm               = mCostFunctionSum.get();
   PVLayerLoc const *bufferLoc = getTargetLayer()->getLayerLoc();
   PVLayerLoc const *maskLoc   = getMaskLayer() ? getMaskLayer()->getLayerLoc() : nullptr;
   int nbatch                  = static_cast<int>(values.size());
   for (int b = 0; b < nbatch; ++b) {
      values.getValue(b) =
            norm->calculateSum(getTargetBuffer(), bufferLoc, getMaskBuffer(), maskLoc, b);
   }
}

template <class C>
void NormProbeLocalTemplate<C>::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   NormProbeLocalInterface::initialize(paramsIO);
}

template <class C>
void NormProbeLocalTemplate<C>::initializeState(HyPerLayer *targetLayer) {
   NormProbeLocalInterface::initializeState(targetLayer);
   mCostFunctionSum = createCostFunctionSum();
}

} // namespace PV

#endif // NORMPROBELOCALTEMPLATE_HPP_
