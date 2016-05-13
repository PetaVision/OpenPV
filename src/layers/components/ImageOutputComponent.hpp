#pragma once

#include "FileOutputComponent.hpp"

#include <string>

namespace PV
{
	class FileOutputComponent;
	
	class ImageOutputComponent : public FileOutputComponent
	{		
		public:
			virtual void initialize();
			//virtual void transform();
			virtual void readFile(std::string fileName) {}
			//virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
		
		protected:
			//virtual void ioParam_fileName(enum ParamsIOFlag ioFlag);
	};
	
	BaseObject * createImageOutputComponent(char const * name, HyPerCol * hc);
}


