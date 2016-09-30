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
   BaseConnectionProbe(const char *probeName, HyPerCol *hc);
   virtual ~BaseConnectionProbe();

   virtual int communicateInitInfo();

   BaseConnection *getTargetConn() { return targetConn; }

  protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char *probeName, HyPerCol *hc);
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();
   int setTargetConn(const char *connName);

   // Member Variables
  protected:
   BaseConnection *targetConn; // The connection itself.
};

} // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
