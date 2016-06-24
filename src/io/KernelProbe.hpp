/*
 * KernelPatchProbe.hpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#ifndef KERNELPROBE_HPP_
#define KERNELPROBE_HPP_

#include "BaseHyPerConnProbe.hpp"

namespace PV {

class KernelProbe : public BaseHyPerConnProbe {

// Methods
public:
   KernelProbe(const char * probename, HyPerCol * hc);
   virtual ~KernelProbe();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int outputState(double timef);
protected:
   KernelProbe(); // Default constructor, can only be called by derived classes
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kernelIndex(enum ParamsIOFlag ioFlag);
   virtual void ioParam_arborId(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPlasticIncr(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPatchIndices(enum ParamsIOFlag ioFlag);
   int patchIndices(HyPerConn * conn);
   
   virtual int initNumValues();
      
   virtual int calcValues(double timevalue) {return PV_FAILURE;}

   int getKernelIndex() {return kernelIndex;}
   int getArbor()       {return arborID;}
   bool getOutputWeights() {return outputWeights;}
   bool getOutputPlasticIncr() {return outputPlasticIncr;}
   bool getOutputPatchIndices() {return outputPatchIndices;}

private:
   int initialize_base();

// Member variables
private:
   int kernelIndex; // which kernel index to investigate
   int arborID; // which arbor to investigate
   bool outputWeights;      // whether to output W
   bool outputPlasticIncr;  // whether to output dW
   bool outputPatchIndices; // whether to output which presynaptic neurons using the given kernel index

}; // end of class KernelProbe block

BaseObject * createKernelProbe(char const * name, HyPerCol * hc);

}  // end of namespace PV block


#endif /* KERNELPROBE_HPP_ */
