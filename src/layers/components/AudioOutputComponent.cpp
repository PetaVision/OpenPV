#include "AudioOutputComponent.hpp"

#include <ifstream>

namespace PV
{
	BaseObject * createAudioOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new AudioOutputComponent() : NULL;
	}
	
	void AudioOutputComponent::updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer)
   {
      if(fileName == "") return; //TODO: This will not return when we're sliding our audio file
      
      std::ifstream wav(fileName, std::ios::in | std::ifstream::binary);
      if(wav == nullptr) { pvError() << mParentLayer->getName() << ": Could not find file << " << fileName << std::endl; throw; }
      
      
      WaveHeader header;
      std::vector<uint8_t> raw;
      std::vector<float> audio;
      wav.read(&header, sizeof(WaveHeader));
      
      if(validHeader(&header))
      {
         int sampleSize = header.bitsPerSample / 8;
         raw.resize(header.dataLength);
         wav.read(&raw[0], header.dataLength);
         
         //TODO: Loop here and actually assemble our audio stream
      }
   }
   
   bool AudioOutputComponent::validHeader(const WavHeader *header)
   {
      bool valid = true;
      //Make sure our header has the correct magic strings
      valid &= (strncmp(header->RIFF, "RIFF", 4) == 0);
      valid &= (strncmp(header->WAVE, "WAVE", 4) == 0);
      valid &= (strncmp(header->FMT,  "FMT ", 4) == 0);
      valid &= (strncmp(header->DATA, "DATA", 4) == 0);
      //We're only reading PCM
      valid &= (header->audioFormat == 1);
      return valid;
   }
}

