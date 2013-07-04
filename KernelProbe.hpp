/*
 * KernelPatchProbe.hpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#ifndef KERNELPROBE_HPP_
#define KERNELPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/KernelConn.hpp"

namespace PV {

class KernelProbe : public BaseConnectionProbe {

// Methods
public:
   KernelProbe(const char * probename, const char * filename, HyPerConn * conn, int kernelIndex, int arborID);
   virtual ~KernelProbe();
   virtual int outputState(double timef);
protected:
   KernelProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, const char * filename, HyPerConn * conn, int kernel, int arbor);
   int patchIndices(KernelConn * kconn);

   KernelConn * getTargetKConn() {return targetKConn;}

private:
   int initialize_base();

// Member variables
protected:
   int kernelIndex; // which kernel index to investigate
   int arborID; // which arbor to investigate
   bool outputWeights;      // whether to output W
   bool outputPlasticIncr;  // whether to output dW
   bool outputPatchIndices; // whether to output which presynaptic neurons using the given kernel index

private:
   KernelConn * targetKConn; // dynamic cast of targetConn to a KernelConn


}; // end of class KernelProbe block

}  // end of namespace PV block


#endif /* KERNELPROBE_HPP_ */
