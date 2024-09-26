/*
 * IndexInternalState.hpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 */

#ifndef INDEXINTERNALSTATE_HPP_
#define INDEXINTERNALSTATE_HPP_

#include <components/InternalStateBuffer.hpp>

namespace PV {

/**
 * A InternalState class meant to be useful in testing.
 * At time t, the value of global batch element b, global restricted index k,
 * is t*(b*N+k), where N is the number of neurons in global restrictes space.
 *
 */
class IndexInternalState : public InternalStateBuffer {
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;

  public:
   IndexInternalState(char const *name, PVParams *params, Communicator const *comm);
   ~IndexInternalState();

  protected:
   IndexInternalState();
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   virtual void updateBufferCPU(double simTime, double deltaTime) override;
};

} // end namespace PV

#endif // INDEXINTERNALSTATE_HPP_
