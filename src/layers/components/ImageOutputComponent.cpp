#include "ImageOutputComponent.hpp"

namespace PV
{
	BaseObject * createImageOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new ImageOutputComponent() : NULL;
	}
	
   void ImageOutputComponent::initialize() { FileOutputComponent::initialize(); }
	
}

