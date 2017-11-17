/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "BaseConnection.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class IdentConn : public BaseConnection {
  public:
   IdentConn(const char *name, HyPerCol *hc);
   virtual int deliver() override;

  protected:
   IdentConn();
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);

#ifdef PV_USE_CUDA
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
#endif // PV_USE_CUDA
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createDeliveryObject() override;

   virtual int setInitialValues() override { return PV_SUCCESS; }

   virtual int outputState(double timestamp) override { return PV_SUCCESS; }
   virtual int updateState(double time, double dt) override { return PV_SUCCESS; }
   virtual bool needUpdate(double time, double dt) override { return false; }
}; // class IdentConn

} // end of block for namespace PV

#endif /* IDENTCONN_HPP_ */
