/*
 * InitRandomWeights.hpp
 *
 *  Created on: Aug 21, 2013
 *      Author: pschultz
 */

#ifndef INITRANDOMWEIGHTS_HPP_
#define INITRANDOMWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "../utils/cl_random.h"

namespace PV {

class InitRandomWeights: public PV::InitWeights {
public:
   InitRandomWeights();
   virtual int calcWeights(/* PVPatch * patch */pvdata_t * dataStart,
         int patchIndex, int arborId, InitWeightsParams *weightParams);
   virtual ~InitRandomWeights();

protected:
   int initRNGs(HyPerConn * conn, bool isKernel);
   virtual int randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, uint4 * rnd_state) = 0;
   // Subclasses must override randomWeights.
   // patchDataStart is a pointer to the beginning of a data patch.
   // rnd_state is a pointer to the random number generator for that patch (the RNGs are initialized in initRNGs).
   // The method should fill the entire patch (the dimensions are in weightParams) regardless of whether the patch is shrunken.
   // This means that weights on different MPI processes that represent the same physical connection will have the same weight.

private:
   int initialize_base();

// Member variables
protected:
   uint4 * rnd_state; // Array of pseudo-RNGs that use the routines in cl_random.
   // For HyPerConns, there will be an RNG for each presynaptic extended neuron.
   // In MPI, if a presynaptic neuron is represented on more than one process, its RNG should be in the same state in each of these
   // processes.  initRNGs handles that by using the presynaptic global extended index to seed the RNG.
   //
   // For KernelConns, there will be an RNG for each data patch.
   // In MPI, the a given data patch should be in the same state across all MPI processes.
   // initRNGs handles that by using the data patch to seed the RNG.
};

} /* namespace PV */
#endif /* INITRANDOMWEIGHTS_HPP_ */
