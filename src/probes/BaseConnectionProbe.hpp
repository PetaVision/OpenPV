/*
 * BaseConnectionProbe.hpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#ifndef BASECONNECTIONPROBE_HPP_
#define BASECONNECTIONPROBE_HPP_

#include "../connections/BaseConnection.hpp"

enum PatchIDMethod { INDEX_METHOD, COORDINATE_METHOD };

namespace PV {

class BaseConnectionProbe : public BaseProbe {

   // Methods
  public:
   BaseConnectionProbe(const char *name, HyPerCol *hc);
   virtual ~BaseConnectionProbe();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   BaseConnection *getTargetConn() { return targetConn; }

  protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived
   // classes
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;

   /**
    * The root process of each MPIBlock sets the vector of PrintStreams to
    * size one, since all batch elements use the same weights. The output file
    * is the probeOutputFile name, if that is set; otherwise it is the logfile.
    */
   virtual void initOutputStreams(const char *filename, Checkpointer *checkpointer) override;

  private:
   int initialize_base();

   // Member Variables
  protected:
   BaseConnection *targetConn; // The connection itself.
};

} // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
