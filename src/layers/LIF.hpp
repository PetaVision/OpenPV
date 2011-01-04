/*
 * LIF.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: C. Rasmussen
 *
 */

#ifndef LIF_HPP_
#define LIF_HPP_

#include "HyPerLayer.hpp"
#include "../arch/opencl/CLKernel.hpp"
#include "../arch/opencl/pv_uint4.h"
#include "../kernels/LIF_params.h"

namespace PV
{

class LIF: public PV::HyPerLayer
{
public:
   LIF(const char* name, HyPerCol * hc);
   LIF(const char* name, HyPerCol * hc, PVLayerType type);
   virtual ~LIF();

   virtual int updateState(float time, float dt);
#ifdef PV_USE_OPENCL
   virtual int updateStateOpenCL(float time, float dt);
#endif
   virtual int writeState(const char * path, float time);

   int setParams(PVParams * p);

protected:

   bool spikingFlag;    // specifies that layer is spiking
   LIF_params lParams;
   uint4 * rand_state;  // state for random numbers

   pvdata_t * Vth;      // threshhold potential
   pvdata_t * G_E;      // excitatory conductance
   pvdata_t * G_I;      // inhibitory conductance
   pvdata_t * G_IB;

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   // OpenCL buffers
   //
   CLBuffer * clRand;
   CLBuffer * clVth;
   CLBuffer * clG_E;
   CLBuffer * clG_I;
   CLBuffer * clG_IB;
#endif

private:
   virtual int initialize(PVLayerType type);

   int findPostSynaptic(int dim, int maxSize, int col,
	// input: which layer, which neuron
			HyPerLayer *lSource, float pos[],

			// output: how many of our neurons are connected.
			// an array with their indices.
			// an array with their feature vectors.
			int* nNeurons, int nConnectedNeurons[], float *vPos);

};

} // namespace PV

#ifdef __cplusplus
extern "C"
{
#endif


#ifdef __cplusplus
}
#endif

#endif /* LIF_HPP_ */
