#pragma once

#include "OutputComponent.hpp"

namespace PV
{
	class OutputComponent;
	
	class SigmoidOutputComponent : public OutputComponent
	{
		public:
			virtual void transform();
			virtual void derivative();
	};
	
	BaseObject * createSigmoidOutputComponent(char const * name, HyPerCol * hc);
}

