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
#include "LIF2.h"

namespace PV
{

// by default uses LIF2 parameters
typedef LIF2_params LIFParams;

class LIF: public PV::HyPerLayer
{
public:
   LIF(const char* name, HyPerCol * hc);
   LIF(const char* name, HyPerCol * hc, PVLayerType type);

   virtual int updateState(float time, float dt);
#ifdef PV_USE_OPENCL
   virtual int updateStateOpenCL(float time, float dt);
#endif
   virtual int writeState(const char * path, float time);

   int setParams(PVParams * params, LIFParams * p);

protected:

#ifdef PV_USE_OPENCL
   virtual int initializeThreadData();
   virtual int initializeThreadKernels();
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
