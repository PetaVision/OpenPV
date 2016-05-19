#include "OutputComponent.hpp"
#include <utils/PVLog.hpp>
namespace PV
{
	BaseObject * createOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new OutputComponent() : NULL;
	}
	
	void OutputComponent::allocateTransformBuffer()
	{
		mTransformBuffer = pvcube_new(mParentLayer->getLayerLoc(), mParentLayer->getNumExtendedAllBatches());
	}
	
	void OutputComponent::allocateDerivativeBuffer()
	{
		mDerivativeBuffer = pvcube_new(mParentLayer->getLayerLoc(), mParentLayer->getNumExtendedAllBatches());
	}
   
   void OutputComponent::clearActivity()
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
         activity[extendedK] = 0;
		}
	}
	
	void OutputComponent::transform()
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
			
			//Linear transfer function
         activity[extendedK] = membranePotential[k];
		}
	}
	
	void OutputComponent::derivative()
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
			int extendedK = mParentLayer->calcActivityIndex(k);
		
			//Derivative of linear function is 1
			deriv[extendedK] = 1.0f;
		}
	}
	

}
