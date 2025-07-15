/*
 * MaskActivityBuffer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MASKACTIVITYBUFFER_HPP_
#define MASKACTIVITYBUFFER_HPP_

#include "components/ANNActivityBuffer.hpp"

namespace PV {

class MaskActivityBuffer : public ANNActivityBuffer {
  protected:
   virtual void ioParam_maskMethod(ParamsIOSwitch ioSwitch);
   virtual void ioParam_maskLayerName(ParamsIOSwitch ioSwitch);
   virtual void ioParam_featureIdxs(ParamsIOSwitch ioSwitch);

  public:
   enum Method { UNDEFINED, LAYER, INVERT_LAYER, FEATURES, INVERT_FEATURES };

   MaskActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   MaskActivityBuffer();
   virtual ~MaskActivityBuffer();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  protected:
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   std::string mMaskMethod;
   Method mMaskMethodCode = UNDEFINED;
   std::string mMaskLayerName;
   ActivityBuffer *mMaskBuffer = nullptr;
   std::vector<int> mFeatures;
   int mNumSpecifiedFeatures   = 0;

}; // class MaskActivityBuffer

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
