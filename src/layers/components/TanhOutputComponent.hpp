#pragma once

#include "OutputComponent.hpp"

namespace PV
{
	class OutputComponent;
	
	class TanhOutputComponent : public OutputComponent
	{
		public:
			virtual void transform();
			virtual void derivative();
	};
	
	BaseObject * createTanhOutputComponent(char const * name, HyPerCol * hc);
}

