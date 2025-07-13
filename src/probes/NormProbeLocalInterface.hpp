#ifndef NORMPROBELOCALINTERFACE_HPP_
#define NORMPROBELOCALINTERFACE_HPP_

#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "structures/PVLayerLoc.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <cstdlib>
#include <memory>

namespace PV {

class NormProbeLocalInterface : public ProbeComponent {
  protected:
   virtual void ioParam_maskLayerName(ParamsIOSwitch ioSwitch);

  public:
   NormProbeLocalInterface(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~NormProbeLocalInterface();

   void clearStoredValues();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);

   virtual void initializeState(HyPerLayer *targetLayer);
   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch);
   void storeValues(double simTime);

   PVLayerLoc const *getLayerLoc() const { return mTargetLayer->getLayerLoc(); }
   ProbeDataBuffer<double> const &getStoredValues() const { return mStoredValues; }

  protected:
   NormProbeLocalInterface() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

   float const *getMaskBuffer() const { return mMaskBuffer; }
   HyPerLayer *getMaskLayer() { return mMaskLayer; }
   HyPerLayer const *getMaskLayer() const { return mMaskLayer; }
   float const *getTargetBuffer() const { return mTargetBuffer; }
   HyPerLayer *getTargetLayer() { return mTargetLayer; }
   HyPerLayer const *getTargetLayer() const { return mTargetLayer; }

  private:
   virtual void calculateNorms(double simTime, ProbeData<double> &values) const = 0;

   void checkMaskLayerDimensions() const;

   float const *findDataBuffer(HyPerLayer *layer) const;

  private:
   float const *mMaskBuffer   = nullptr;
   HyPerLayer *mMaskLayer     = nullptr;
   std::string mMaskLayerName;
   ProbeDataBuffer<double> mStoredValues;
   float const *mTargetBuffer = nullptr;
   HyPerLayer *mTargetLayer   = nullptr;
};

} // namespace PV

#endif // NORMPROBELOCALINTERFACE_HPP_
