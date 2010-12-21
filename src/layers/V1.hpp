/*
 * V1.h
 *
 *  Created on: Jul 30, 2008
 *
 */

#ifndef V1_HPP_
#define V1_HPP_

#include "HyPerLayer.hpp"
#include "../arch/opencl/CLKernel.hpp"
#include "LIF2.h"

namespace PV
{

// by default uses LIF2 parameters
typedef LIF2_params LIFParams;

class V1: public PV::HyPerLayer
{
public:
   V1(const char* name, HyPerCol * hc);
   V1(const char* name, HyPerCol * hc, PVLayerType type);

   virtual int updateState(float time, float dt);
#ifdef PV_USE_OPENCL
   virtual int updateStateOpenCL(float time, float dt);
#endif
   virtual int writeState(const char * path, float time);

   int setParams(PVParams * params, LIFParams * p);

protected:
   virtual int updateV();
   virtual int setActivity();
   virtual int resetPhiBuffers();
   int resetBuffer(pvdata_t * buf, int numItems);

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

#endif /* V1_HPP_ */
