#include "SigmoidOutputComponent.hpp"

namespace PV
{
	BaseObject * createSigmoidOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new SigmoidOutputComponent() : NULL;
	}
	
	void SigmoidOutputComponent::transform()
	{
		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		int batchOffset = mParentLayer->calcBatchOffset();
		
		#ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for schedule(static)
		#endif
		for(int index = 0; index < numNeurons*numBatches; index++ )
		{
			int b = index / numNeurons;
			int k = index % numNeurons;
			MEM_GLOBAL pvdata_t * activity = mTransformBuffer->data + b * batchOffset;
			MEM_GLOBAL pvdata_t * membranePotential = mParentLayer->getV() + b * numNeurons;
			int extendedK = mParentLayer->calcActivityIndex(k);
			
			//Sigmoid
			activity[extendedK] = 1.0f / (1.0 + exp(-membranePotential[k]));
		}
	}
	
	void SigmoidOutputComponent::derivative()
	{
		if(mDerivativeBuffer == nullptr) return;
		
		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		int batchOffset = mParentLayer->calcBatchOffset();
		
		#ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for schedule(static)
		#endif
		for(int index = 0; index < numNeurons*numBatches; index++ )
		{
			int b = index / numNeurons;
			int k = index % numNeurons;
			MEM_GLOBAL pvdata_t * deriv = mDerivativeBuffer->data + b * batchOffset;
			MEM_GLOBAL pvdata_t * activity = mTransformBuffer->data + b * batchOffset;
			int extendedK = mParentLayer->calcActivityIndex(k);
			
			//Derivative of sigmoid function f(x) is f(x) * (1 - f(x))
			deriv[extendedK] = activity[extendedK] * (1.0 - activity[extendedK]);
		}
	}
}
