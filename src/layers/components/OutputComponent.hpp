#pragma once

#include "Component.hpp"
#include <layers/PVLayerCube.h>

#include <vector>

namespace PV
{
	class OutputComponent : public Component<PVLayerCube>
	{
		public:
			virtual void allocateTransformBuffer();
			virtual void allocateDerivativeBuffer();
			virtual void transform();
			virtual void derivative();
	};
	
	BaseObject * createOutputComponent(char const * name, HyPerCol * hc);
}
