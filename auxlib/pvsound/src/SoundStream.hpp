/*
 * SoundStream.hpp
 *
 *  Created on: May 6, 2013
 *      Author: slundquist 
 */

#ifndef SOUNDSTREAM_HPP_
#define SOUNDSTREAM_HPP_

#include <layers/HyPerLayer.hpp>
#include <columns/HyPerCol.hpp>
#include <stdio.h>
#include <sndfile.h>

//#include "../utils/cl_random.h"

#ifndef STAT_H
#include <sys/stat.h>
#endif

namespace PVsound {

class SoundStream : public PV::HyPerLayer{

public:
   SoundStream(const char * name, PV::HyPerCol * hc);
   virtual ~SoundStream();
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);
   virtual bool activityIsSpiking() { return false; }

private:
   int initialize_base();

protected:
   SoundStream();
   int initialize(const char * name, PV::HyPerCol * hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_soundInputPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_frameStart(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual double getDeltaUpdateTime();
   virtual int setActivity();

   pvdata_t * soundData; //Buffer containing image
   SF_INFO* fileHeader;
   SNDFILE* fileStream;
   float* soundBuf;

   // double displayPeriod;     // Length of time a string 'frame' is displayed
   int frameStart;

    int sampleRate;          // sample rate from file in Hz
   // double nextSampleTime;   // time at which next sample is retrieved
   char * filename;          // Path to file if a file exists

}; // end class SoundStream

PV::BaseObject * createSoundStream(char const * name, PV::HyPerCol * hc);

}  // end namespace PVsound

#endif /* SOUNDSTREAM_HPP_ */
