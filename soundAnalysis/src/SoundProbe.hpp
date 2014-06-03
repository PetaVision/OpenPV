/*
 * SoundProbe.hpp
 * Author: slundquist
 */

#ifndef SOUNDPROBE_HPP_
#define SOUNDPROBE_HPP_

//Only compile this file and its cpp if using sound sandbox
#ifdef PV_USE_SNDFILE

#include <io/StatsProbe.hpp>
#include <sndfile.h>

namespace PV{

class SoundProbe : public PV::StatsProbe{
public:
   SoundProbe(const char * probeName, HyPerCol * hc);
   ~SoundProbe();

   virtual int outputState(double timed);
   virtual int communicateInitInfo();

protected:
   int initSoundProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_soundOutputPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_soundInputType(enum ParamsIOFlag ioFlag);

private:
   int init_base();
   char* soundOutputPath;
   char* soundInputType;
   SF_INFO* fileHeader;
   SNDFILE* fileStream;
   float* soundBuf;

};

}
#endif /* PV_USE_SNDFILE */

#endif 
