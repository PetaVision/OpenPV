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
         virtual double getDeltaUpdateTime() { return -1.0; }
         void clearActivity();
	};
	
	BaseObject * createOutputComponent(char const * name, HyPerCol * hc);
}
