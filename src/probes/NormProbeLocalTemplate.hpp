#ifndef NORMPROBELOCALTEMPLATE_HPP_
#define NORMPROBELOCALTEMPLATE_HPP_

#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "structures/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
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
   NormProbeLocalTemplate(char const *objName, PVParams *params);
   virtual ~NormProbeLocalTemplate() {}

   virtual void initializeState(HyPerLayer *targetLayer) override;

  protected:
   NormProbeLocalTemplate() {}
   virtual std::shared_ptr<C const> createCostFunctionSum() { return nullptr; }
   void initialize(char const *objName, PVParams *params);

  private:
   void calculateNorms(double simTime, ProbeData<double> &values) const override;

  private:
   std::shared_ptr<C const> mCostFunctionSum = nullptr;
};

template <class C>
NormProbeLocalTemplate<C>::NormProbeLocalTemplate(char const *objName, PVParams *params) {
   initialize(objName, params);
}

template <class C>
void NormProbeLocalTemplate<C>::calculateNorms(double simTime, ProbeData<double> &values) const {
   C const *norm               = mCostFunctionSum.get();
   PVLayerLoc const *bufferLoc = getTargetLayer()->getLayerLoc();
   int nbatch                  = static_cast<int>(values.size());
   PVLayerCube targetLayerCube = findLayerCube(getTargetLayer());
   if (getMaskLayer()) {
      PVLayerCube maskLayerCube = findLayerCube(getMaskLayer());
      for (int b = 0; b < nbatch; ++b) {
         float const *targetBuffer = targetLayerCube.data[b]; 
         float const *maskBuffer   = maskLayerCube.data[b]; 
         PVLayerLoc const *maskLoc = getMaskLayer()->getLayerLoc();
         values.getValue(b) =
               norm->calculateSum(b, &targetLayerCube, &maskLayerCube);
      }
   }
   else {
      for (int b = 0; b < nbatch; ++b) {
         float const *targetBuffer = targetLayerCube.data[b]; 
         values.getValue(b) = norm->calculateSum(b, &targetLayerCube, nullptr);
      }
   }
}

template <class C>
void NormProbeLocalTemplate<C>::initialize(char const *objName, PVParams *params) {
   NormProbeLocalInterface::initialize(objName, params);
}

template <class C>
void NormProbeLocalTemplate<C>::initializeState(HyPerLayer *targetLayer) {
   NormProbeLocalInterface::initializeState(targetLayer);
   mCostFunctionSum = createCostFunctionSum();
}

} // namespace PV

#endif // NORMPROBELOCALTEMPLATE_HPP_
