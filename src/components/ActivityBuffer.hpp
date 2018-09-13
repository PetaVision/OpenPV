/*
 * ActivityBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef ACTIVITYBUFFER_HPP_
#define ACTIVITYBUFFER_HPP_

#include "components/BufferComponent.hpp"
#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A component to contain the activity buffer of a HyPerLayer.
 */
class ActivityBuffer : public BufferComponent {

  public:
   ActivityBuffer(char const *name, HyPerCol *hc);

   virtual ~ActivityBuffer();

   virtual void updateState(double simTime, double deltaTime) override;

   float *getActivity() { return mBufferData.data(); }

  protected:
   ActivityBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   void setActivity();

  private:
   //    void checkDimensions(int internalStateSize, int activitySize, char const *fieldname);

  protected:
   InternalStateBuffer *mInternalState = nullptr;
};

} // namespace PV

#endif // ACTIVITYBUFFER_HPP_
