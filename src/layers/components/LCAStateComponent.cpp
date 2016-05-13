#include "LCAStateComponent.hpp"
#include <utils/PVLog.hpp>

namespace PV
{
	BaseObject * createLCAStateComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new LCAStateComponent() : NULL;
	}
   
   LCAStateComponent::~LCAStateComponent()
   {
      if(mPreviousTransformBuffer != nullptr) delete mPreviousTransformBuffer;
   }
   
   void LCAStateComponent::allocateTransformBuffer()
   {
      StateComponent::allocateTransformBuffer();
		
      mPreviousTransformBuffer = (pvdata_t*)calloc(mParentLayer->getNumNeuronsAllBatches(), sizeof(pvdata_t));
      if(mPreviousTransformBuffer == NULL)
		{
			fprintf(stderr, "Layer \"%s\" error in rank %d process: unable to allocate memory for %s: %s.\n",
				mParentLayer->getName(), mParentLayer->getParent()->columnId(), "previous membrane potential V", strerror(errno));
			throw;
		}
   }
   
   void LCAStateComponent::initialize()
	{
      int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
      
      #ifdef PV_USE_OPENMP_THREADS
         #pragma omp parallel for schedule(static)
      #endif
      for (int index = 0; index < numNeurons*numBatches; index++)
      {
         int b = index / numNeurons;
         int k = index % numNeurons;

         MEM_GLOBAL pvdata_t* currentState = mTransformBuffer + b * numNeurons;
         MEM_GLOBAL pvdata_t* previousState = mPreviousTransformBuffer + b * numNeurons;
         previousState[k] = currentState[k];
      }
   }
	
	void LCAStateComponent::transform()
	{
		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		int batchOffset = mParentLayer->calcBatchOffset();
      double *dtAdapt = mParentLayer->getParent()->getTimeScale();
      
      #ifdef PV_USE_OPENMP_THREADS
         #pragma omp parallel for schedule(static)
      #endif
      for (int index = 0; index < numNeurons*numBatches; index++)
      {
         int b = index / numNeurons;
         int k = index % numNeurons;
         float expTau = exp(-dtAdapt[b] / mTimeConstantTau);
         
         MEM_GLOBAL pvdata_t* currentState = mTransformBuffer + b * numNeurons;
         MEM_GLOBAL pvdata_t* previousState = mPreviousTransformBuffer + b * numNeurons;
         MEM_GLOBAL pvdata_t* activityBatch = mParentLayer->getActivity() + b * batchOffset;
         
         currentState[k] = expTau * previousState[k] + (1.0f - expTau) * (currentState[k] + mSelfInteract * activityBatch[mParentLayer->calcActivityIndex(k)]);
         previousState[k] = currentState[k];
      }
   }

  	void LCAStateComponent::derivative() {}

	void LCAStateComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
	{
		ioParam_timeConstantTau(ioFlag);
		ioParam_selfInteract(ioFlag);
   }
   
   void LCAStateComponent::ioParam_timeConstantTau(enum ParamsIOFlag ioFlag)
   {
      parent->ioParamValue(ioFlag, mParentLayer->getName(), "timeConstantTau", &mTimeConstantTau, mTimeConstantTau, true);
   }

   void LCAStateComponent::ioParam_selfInteract(enum ParamsIOFlag ioFlag)
   {
      parent->ioParamValue(ioFlag, mParentLayer->getName(), "selfInteract", &mSelfInteract, mSelfInteract);
   }
}
