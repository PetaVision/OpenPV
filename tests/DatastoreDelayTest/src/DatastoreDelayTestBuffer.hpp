/*
 * DatastoreDelayTest.hpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#ifndef DATASTOREDELAYTESTBUFFER_HPP_
#define DATASTOREDELAYTESTBUFFER_HPP_

#include <components/InternalStateBuffer.hpp>
#include <layers/HyPerLayer.hpp> // Temporary hack - getNumDelayLevels isn't in a component yet.

namespace PV {

class DatastoreDelayTestBuffer : public InternalStateBuffer {

  public:
   DatastoreDelayTestBuffer(const char *name, HyPerCol *hc);
   virtual ~DatastoreDelayTestBuffer();

  protected:
   int initialize(const char *name, HyPerCol *hc);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   void updateState(
         double simTime,
         double dt,
         int numNeurons,
         float *V,
         float *A,
         int nx,
         int ny,
         int nf,
         int lt,
         int rt,
         int dn,
         int up);

   static int updateV(const PVLayerLoc *loc, bool *inited, float *V, int period);

  protected:
   bool inited;
   int mPeriod;

}; // end of class DatastoreDelayTestBuffer block

} // end of namespace PV block

#endif /* DATASTOREDELAYTESTBUFFER_HPP_ */
