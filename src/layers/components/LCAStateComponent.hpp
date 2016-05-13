#pragma once

#include "StateComponent.hpp"

namespace PV
{
	class StateComponent;
	
	class LCAStateComponent : public StateComponent
	{
		public:
			~LCAStateComponent();
			virtual void allocateTransformBuffer();
			virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
			virtual void initialize();
			virtual void transform();
			virtual void derivative();
			
		protected:
			pvdata_t *mPreviousTransformBuffer = nullptr;
			pvdata_t mTimeConstantTau = 1.0f;
			bool mSelfInteract = true;
			
			virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
			virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);
	};
	
	BaseObject * createLCAStateComponent(char const * name, HyPerCol * hc);
}

