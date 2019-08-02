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
   virtual void ioParam_maskMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_featureIdxs(enum ParamsIOFlag ioFlag);

  public:
   enum Method { UNDEFINED, LAYER, INVERT_LAYER, FEATURES, INVERT_FEATURES };

   MaskActivityBuffer(const char *name, PVParams *params, Communicator const *comm);
   MaskActivityBuffer();
   virtual ~MaskActivityBuffer();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;

  protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   char *mMaskMethod           = nullptr;
   Method mMaskMethodCode      = UNDEFINED;
   char *mMaskLayerName        = nullptr;
   ActivityBuffer *mMaskBuffer = nullptr;
   int *mFeatures              = nullptr;
   int mNumSpecifiedFeatures   = 0;

}; // class MaskActivityBuffer

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
