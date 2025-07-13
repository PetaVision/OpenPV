/*
 * ShrunkenPatchTestActivityBuffer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef SHRUNKENPATCHTESTACTIVITYBUFFER_HPP_
#define SHRUNKENPATCHTESTACTIVITYBUFFER_HPP_

#include "components/ActivityBuffer.hpp"

namespace PV {

/**
 * ShrunkenPatchTestActivityBuffer is the ActivityBuffer subclass for MPITestLayer
 */
class ShrunkenPatchTestActivityBuffer : public ActivityBuffer {
  public:
   ShrunkenPatchTestActivityBuffer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~ShrunkenPatchTestActivityBuffer();

  protected:
   ShrunkenPatchTestActivityBuffer() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   void setActivityToGlobalPos();
};

} // namespace PV

#endif // SHRUNKENPATCHTESTACTIVITYBUFFER_HPP_
