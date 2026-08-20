#ifndef NORMPROBELOCALINTERFACE_HPP_
#define NORMPROBELOCALINTERFACE_HPP_

#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "structures/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
#include "layers/BaseLayer.hpp"
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
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);

  public:
   NormProbeLocalInterface(char const *objName, PVParams *params);
   virtual ~NormProbeLocalInterface();

   void clearStoredValues();

   Response::Status communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message);

   virtual void initializeState(BaseLayer *targetLayer);
   virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void storeValues(double simTime);

   PVLayerLoc const *getLayerLoc() const { return mTargetLayer->getLayerLoc(); }
   ProbeDataBuffer<double> const &getStoredValues() const { return mStoredValues; }

  protected:
   NormProbeLocalInterface() {}
   void initialize(char const *objName, PVParams *params);

   float const *getMaskBuffer() const { return mMaskBuffer; }
   BaseLayer *getMaskLayer() { return mMaskLayer; }
   BaseLayer const *getMaskLayer() const { return mMaskLayer; }
   float const *getTargetBuffer() const { return mTargetBuffer; }
   BaseLayer *getTargetLayer() { return mTargetLayer; }
   BaseLayer const *getTargetLayer() const { return mTargetLayer; }

  private:
   virtual void calculateNorms(double simTime, ProbeData<double> &values) const = 0;

   void checkMaskLayerDimensions() const;

   float const *findDataBuffer(BaseLayer *layer) const;

  private:
   float const *mMaskBuffer = nullptr;
   BaseLayer *mMaskLayer   = nullptr;
   char *mMaskLayerName     = nullptr;
   ProbeDataBuffer<double> mStoredValues;
   float const *mTargetBuffer = nullptr;
   BaseLayer *mTargetLayer   = nullptr;
};

} // namespace PV

#endif // NORMPROBELOCALINTERFACE_HPP_
