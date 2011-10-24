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
   KernelProbe(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborID);
   virtual ~KernelProbe();
   virtual int outputState(float time, HyPerConn * c);
protected:
   KernelProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, const char * filename, HyPerCol * hc, int kernel, int arbor);
   int patchIndices(KernelConn * kconn);

private:
   int initialize_base();

// Member variables
protected:
   int kernelIndex; // which kernel index to investigate
   int arborID; // which arbor to investigate
   bool outputWeights;      // whether to output W
   bool outputPlasticIncr;  // whether to output dW
   bool outputPatchIndices; // whether to output which presynaptic neurons using the given kernel index


}; // end of class KernelProbe block

}  // end of namespace PV block


#endif /* KERNELPROBE_HPP_ */
