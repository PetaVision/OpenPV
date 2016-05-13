#include "StateComponent.hpp"
#include <utils/PVLog.hpp>

namespace PV
{
	BaseObject * createStateComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new StateComponent() : NULL;
	}
	
	void StateComponent::allocateTransformBuffer()
	{
		mTransformBuffer = (pvdata_t*)calloc(mParentLayer->getNumNeuronsAllBatches(), sizeof(pvdata_t));
		if(mTransformBuffer == NULL)
		{
			fprintf(stderr, "Layer \"%s\" error in rank %d process: unable to allocate memory for %s: %s.\n",
				mParentLayer->getName(), mParentLayer->getParent()->columnId(), "membrane potential V", strerror(errno));
			throw;
		}
	}
}
