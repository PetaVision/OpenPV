/*
 * GaussianNoiseActivityBuffer.hpp
 *
 *  Created on: Jul 11, 2019
 *      Author: jspringer
 */

#ifndef GAUSSIANNOISEACTIVITYBUFFER_HPP_
#define GAUSSIANNOISEACTIVITYBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"
#include "components/HyPerActivityBuffer.hpp"

#include <cstdlib>
#include <random>

namespace PV {

class GaussianNoiseActivityBuffer : public HyPerActivityBuffer {

  public:
   GaussianNoiseActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~GaussianNoiseActivityBuffer();

  protected:
   GaussianNoiseActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sigma(enum ParamsIOFlag ioFlag);
   virtual void ioParam_mu(enum ParamsIOFlag ioFlag);

   virtual void setObjectType() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   std::default_random_engine mGenerator;
   std::normal_distribution<float> mDistribution;

   float mMu = 0.0f;
   float mSigma = 1.0f;
};

} // namespace PV

#endif // GAUSSIANNOISEACTIVITYBUFFER_HPP_
