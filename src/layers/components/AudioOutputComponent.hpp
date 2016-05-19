#pragma once

#include "FileOutputComponent.hpp"

#include <string>
#include <cstdint>

namespace PV
{
	class FileOutputComponent;
	
	class AudioOutputComponent : public FileOutputComponent
	{		
		public:
			virtual void updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer);
         
         
      private:
         bool validHeader(WavHeader header);
      
         struct WavHeader
         {
            // Wave header based on:
            // http://soundfile.sapp.org/doc/WaveFormat/
            
            //Riff chunk
            uint8_t     RIFF[4];       // 0x52494646 "RIFF"
            uint32_t    riffChunkSize;
            uint8_t     WAVE[4];       // 0x57415645 "WAVE"
            
            //Fmt info chunk
            uint8_t     FMT[4];        // 0x666d7420 "FMT "
            uint32_t    fmtChunkSize;
            uint16_t    audioFormat;
            uint16_t    channelCount;
            uint32_t    sampleRate;
            uint32_t    bytesPerSecond;
            uint16_t    blockAlign;    // 2 (16 bit mono), 4 (16 bit stereo). Channels are interleved every blockAlign / channelCount samples
            uint16_t    bitsPerSample;
            
            //Data info chunk
            uint8_t     DATA[4];       // 0x64617461 "DATA"
            uint32_t    dataLength;
         };
	};
	
	BaseObject * createAudioOutputComponent(char const * name, HyPerCol * hc);
}


