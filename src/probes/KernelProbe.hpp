/*
 * KernelPatchProbe.hpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#ifndef KERNELPROBE_HPP_
#define KERNELPROBE_HPP_

#include "BaseHyPerConnProbe.hpp"
#include "components/PatchSize.hpp"

namespace PV {

class KernelProbe : public BaseHyPerConnProbe {

   // Methods
  public:
   KernelProbe(const char *probename, PVParams *params, Communicator *comm);
   virtual ~KernelProbe();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status outputState(double simTime, double deltaTime) override;

   PatchSize const *getPatchSize() const { return mPatchSize; }
   Weights const *getWeights() const { return mWeights; }
   float const *getWeightData() const { return mWeightData; }
   float const *getDeltaWeightData() const { return mDeltaWeightData; }

  protected:
   KernelProbe(); // Default constructor, can only be called by derived classes
   void initialize(const char *probename, PVParams *params, Communicator *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_kernelIndex(enum ParamsIOFlag ioFlag);
   virtual void ioParam_arborId(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPlasticIncr(enum ParamsIOFlag ioFlag);
   virtual void ioParam_outputPatchIndices(enum ParamsIOFlag ioFlag);
   int patchIndices();

   virtual void initNumValues() override;

   virtual void calcValues(double timevalue) override {
      Fatal().printf("%s does not use calcValues.\n", getDescription_c());
   };

   int getKernelIndex() { return kernelIndex; }
   int getArbor() { return arborID; }
   bool getOutputWeights() { return outputWeights; }
   bool getOutputPlasticIncr() { return outputPlasticIncr; }
   bool getOutputPatchIndices() { return outputPatchIndices; }

  private:
   int initialize_base();

   // Member variables
  private:
   int kernelIndex; // which kernel index to investigate
   int arborID; // which arbor to investigate
   bool outputWeights; // whether to output W
   bool outputPlasticIncr; // whether to output dW
   bool outputPatchIndices; // whether to output which presynaptic neurons using
   // the given kernel index

   PatchSize const *mPatchSize;
   float const *mWeightData;
   float const *mDeltaWeightData;

}; // end of class KernelProbe block

} // end of namespace PV block

#endif /* KERNELPROBE_HPP_ */
