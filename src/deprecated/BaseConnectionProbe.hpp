/*
 * BaseConnectionProbe.hpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#ifndef BASECONNECTIONPROBE_HPP_
#define BASECONNECTIONPROBE_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "probes/BaseProbe.hpp"

enum PatchIDMethod { INDEX_METHOD, COORDINATE_METHOD };

namespace PV {

class BaseConnectionProbe : public BaseProbe {

   // Methods
  public:
   BaseConnectionProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~BaseConnectionProbe();

   ComponentBasedObject *getTargetConn() { return mTargetConn; }

  protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived
   // classes
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;

   Response::Status respondConnectionProbeWriteParams(
         std::shared_ptr<ConnectionProbeWriteParamsMessage const> message);

   Response::Status respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * The root process of each MPIBlock sets the vector of PrintStreams to
    * size one, since all batch elements use the same weights. The output file
    * is the probeOutputFile name, if that is set; otherwise it is the logfile.
    */
   virtual void initOutputStreams(
        std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   // Member Variables
  protected:
   ComponentBasedObject *mTargetConn = nullptr; // The connection being probed.
   Timer *mIOTimer                   = nullptr;
};

} // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
