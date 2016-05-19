#include "AudioOutputComponent.hpp"


namespace PV
{
	BaseObject * createAudioOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new AudioOutputComponent() : NULL;
	}
	
	void AudioOutputComponent::updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer)
   {
      
   }
}

