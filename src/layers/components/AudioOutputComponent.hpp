#pragma once

#include "FileOutputComponent.hpp"

#include <string>

namespace PV
{
	class FileOutputComponent;
	
	class AudioOutputComponent : public FileOutputComponent
	{		
		public:
			virtual void updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer);
	};
	
	BaseObject * createAudioOutputComponent(char const * name, HyPerCol * hc);
}


