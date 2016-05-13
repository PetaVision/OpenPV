#pragma once

#include "Component.hpp"
#include <connections/BaseConnection.hpp>
#include <include/pv_datatypes.h>

#include <vector>

namespace PV
{
	using ::std::vector;
	
	class InputComponent : public Component<pvdata_t*>
	{
		public:
			void resetBuffer(double time, double dt);
			virtual void receive(vector<BaseConnection*> *connections);
			virtual void allocateTransformBuffer();
			virtual void allocateDerivativeBuffer() {}
			virtual void transform();
			virtual void derivative() {};
	};
	
	BaseObject * createInputComponent(char const * name, HyPerCol * hc);
}
