/*
 * BaseConnectionProbe.hpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#ifndef BASECONNECTIONPROBE_HPP_
#define BASECONNECTIONPROBE_HPP_

#include "../connections/HyPerConn.hpp"

namespace PV {

class BaseConnectionProbe {

// Methods
public:
   virtual ~BaseConnectionProbe();
   virtual int outputState(float time, HyPerConn * c) = 0;
protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, const char * filename, HyPerCol * hc);

private:
   int initialize_base();

// Member variables
protected:
   char * name; // Name of the probe; corresponds to the group name in the params file
   char * filename; // Name of the output file.  Can be NULL if output goes to stdout
   FILE * fp; // pointer to output file; NULL except for root process.  If filename is NULL, fp will be stdout.

}; // end of class BaseConnectionProbe block

}  // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
