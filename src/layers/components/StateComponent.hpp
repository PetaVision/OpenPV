#pragma once

#include "Component.hpp"
#include <include/pv_datatypes.h>

namespace PV
{
	class StateComponent : public Component<pvdata_t>
	{
		public:
			virtual void allocateTransformBuffer();
			virtual void allocateDerivativeBuffer() {}
			virtual void transform() {}
			virtual void derivative() {}
	};
	
	BaseObject * createStateComponent(char const * name, HyPerCol * hc);
}
