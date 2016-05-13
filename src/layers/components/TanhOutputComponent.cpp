#include "TanhOutputComponent.hpp"

namespace PV
{
	BaseObject * createTanhOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new TanhOutputComponent() : NULL;
	}
	
	void TanhOutputComponent::transform()
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
			
			//tanh
         float ePos = exp(membranePotential[k]);
			float eNeg = exp(-membranePotential[k]);
			activity[extendedK] = (ePos - eNeg) / (ePos + eNeg);
		}
	}
	
	void TanhOutputComponent::derivative()
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
			
			//Derivative of tanh function f(x) is 1 - f(x)^2
			deriv[extendedK] = 1.0 - activity[extendedK]*activity[extendedK];
		}
	}
}
