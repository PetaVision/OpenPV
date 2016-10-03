/*
 * InitRandomWeights.hpp
 *
 *  Created on: Aug 21, 2013
 *      Author: pschultz
 */

#ifndef INITRANDOMWEIGHTS_HPP_
#define INITRANDOMWEIGHTS_HPP_

#include "../columns/Random.hpp"
#include "InitWeights.hpp"

namespace PV {

class InitRandomWeights : public PV::InitWeights {
  public:
   virtual int calcWeights(/* PVPatch * patch */ pvdata_t *dataStart, int patchIndex, int arborId);
   virtual ~InitRandomWeights();

  protected:
   InitRandomWeights();
   int initialize(char const *name, HyPerCol *hc);
   virtual int initRNGs(bool isKernel);
   virtual int
   randomWeights(pvdata_t *patchDataStart, InitWeightsParams *weightParams, int patchIndex) = 0;
   // Subclasses must implement randomWeights.
   // patchDataStart is a pointer to the beginning of a data patch.
   // patchIndex is the index for that patch.  The RNGs are accessed by calling randState's get
   // method;
   //     or random numbers are generated directly by calling randState methods.
   // The method should fill the entire patch (the dimensions are in weightParams) regardless of
   // whether the patch is shrunken.
   // This means that weights on different MPI processes that represent the same physical synapse
   // will have the same weight.

  private:
   int initialize_base();

   // Member variables
  protected:
   Random *randState;
   // For HyPerConns, there will be an RNG for each presynaptic extended neuron.
   // In MPI, if a presynaptic neuron is represented on more than one process, its RNG should be in
   // the same state in each of these
   // processes.  initRNGs handles that by using the presynaptic global extended index to seed the
   // RNG.
   //
   // For KernelConns, there will be an RNG for each data patch.
   // In MPI, the a given data patch should be in the same state across all MPI processes.
   // initRNGs handles that by using the data patch to seed the RNG.
};

} /* namespace PV */
#endif /* INITRANDOMWEIGHTS_HPP_ */
