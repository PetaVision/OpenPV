/*
 * V1.h
 *
 *  Created on: Jul 30, 2008
 *
 */

#ifndef V1_H_
#define V1_H_

#include "HyPerLayer.hpp"
#include "LIF2.h"

namespace PV
{

// by default uses LIF2 parameters
typedef LIF2_params V1Params;

class V1: public PV::HyPerLayer
{
public:
   V1(const char* name, HyPerCol * hc);
   V1(const char* name, HyPerCol * hc, PVLayerType type);

   virtual int updateState(float time, float dt);
   virtual int writeState(const char * path, float time);

   int setParams(PVParams * params, V1Params * p);

private:
   void initialize(const char* name, HyPerCol * hc, PVLayerType type);

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

#endif /* V1_H_ */
