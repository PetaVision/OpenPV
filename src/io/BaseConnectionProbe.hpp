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
   virtual int outputState(float timef) = 0;
   const char * getName()               {return name;}
   const char * getFilename()           {return filename;}
   FILE * getFilePtr()                  {return fp;}
   HyPerConn * getTargetConn()          {return targetConn;}

protected:
   BaseConnectionProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, const char * filename, HyPerConn * conn);

private:
   int initialize_base();

// Member variables
private:
   char * name; // Name of the probe; corresponds to the group name in the params file
   char * filename; // Name of the output file.  Can be NULL if output goes to stdout
   FILE * fp; // pointer to output file; NULL except for root process.  If filename is NULL, fp will be stdout.
   HyPerConn * targetConn;

}; // end of class BaseConnectionProbe block

}  // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
