/*
 * BaseConnectionProbe.hpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#ifndef BASECONNECTIONPROBE_HPP_
#define BASECONNECTIONPROBE_HPP_

#include "../connections/HyPerConn.hpp"

enum PatchIDMethod { INDEX_METHOD, COORDINATE_METHOD };

namespace PV {

class BaseConnectionProbe {

// Methods
public:
   virtual ~BaseConnectionProbe();
   virtual int communicate();
   virtual int allocateProbe();
   virtual int outputState(double timed) = 0;

   const char * getName()                  {return name;}
   const char * getTargetConnName()        {return targetConnName;}
   HyPerConn * getTargetConn()             {return targetConn;}

protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, HyPerCol * hc);
   int initialize_base();

   HyPerCol * getParent()                  {return parent;}
   PV_Stream * getStream()                 {return stream;}

private:

// Member Variables
protected:
   HyPerCol * parent; // HyPerCol that owns the probe
   char * name; // Name of the probe; corresponds to the group name in the params file
   PV_Stream * stream; // pointer to output file; NULL except for root process.  If filename is NULL, fp will be stdout.
   char * targetConnName; // The name of the connection being probed.
   HyPerConn * targetConn; // The connection itself.
};

}  // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
