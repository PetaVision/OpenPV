#include "InputComponent.hpp"

#include <connections/HyPerConn.hpp>
#include <utils/PVLog.hpp>

namespace PV
{
	BaseObject * createInputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new InputComponent() : NULL;
	}
	
	void InputComponent::resetBuffer(double time, double dt)
	{
		int nBatch = mParentLayer->getParent()->getNBatch();
		int numNeurons = mParentLayer->getNumNeurons();
		int numChannels = mParentLayer->getNumChannels();
		
		for(int channel = 0; channel < numChannels; channel++)
		{
			MEM_GLOBAL pvdata_t * channelStart = mTransformBuffer[0] + channel * nBatch * numNeurons;
			
         #ifdef PV_USE_OPENMP_THREADS
				#pragma omp parallel for schedule(static)
			#endif
			for(int k = 0; k < numNeurons * nBatch; k++)
			{
				channelStart[k] = 0.0f;
			}
		}
	}
	
	void InputComponent::receive(vector<BaseConnection*> *connections)
	{
		for(vector<BaseConnection*>::iterator it = connections->begin(); it < connections->end(); it++)
		{
			HyPerConn * connection = dynamic_cast<HyPerConn*>(*it);
			connection->deliver();
		}
	}
	
	void InputComponent::allocateTransformBuffer()
	{
		int numChannels = mParentLayer->getNumChannels();
		if (numChannels > 0)
		{
			mTransformBuffer = (pvdata_t **)malloc(numChannels*sizeof(pvdata_t *));
			if(mTransformBuffer == nullptr) { throw; }
			mTransformBuffer[0] = (pvdata_t*)calloc(mParentLayer->getNumNeuronsAllBatches() * numChannels, sizeof(pvdata_t));
			if(mTransformBuffer[0] == nullptr) { throw; }
			for (int m = 1; m < numChannels; m++)
			{
				mTransformBuffer[m] = mTransformBuffer[0] + m * mParentLayer->getNumNeuronsAllBatches();
			}
		}
	}
	
	void InputComponent::transform()
	{
		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		
		MEM_GLOBAL pvdata_t * GSynExc = mTransformBuffer[0] + CHANNEL_EXC * numBatches * numNeurons;
		MEM_GLOBAL pvdata_t * GSynInh = mTransformBuffer[0] + CHANNEL_INH * numBatches * numNeurons;

		#ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for schedule(static)
		#endif
      for(int index = 0; index < numNeurons * numBatches; index++)
		{
			int b = index / numNeurons;
			int k = index % numNeurons;
			
			MEM_GLOBAL pvdata_t* state = mParentLayer->getV() + b*numNeurons;
			MEM_GLOBAL pvdata_t* excInput = GSynExc + b*numNeurons;
			MEM_GLOBAL pvdata_t* inhInput = GSynInh + b*numNeurons;

			state[k] = excInput[k] - inhInput[k];
		}
	}
}
