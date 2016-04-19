/*
 * SoundProbe.hpp
 * Author: slundquist
 */

#ifndef SOUNDPROBE_HPP_
#define SOUNDPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <sndfile.h>

class SoundProbe : public PV::StatsProbe{
public:
   SoundProbe(const char * probeName, PV::HyPerCol * hc);
   ~SoundProbe();

   virtual int outputState(double timed);
   virtual int communicateInitInfo();

protected:
   int initSoundProbe(const char * probeName, PV::HyPerCol * hc);
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
    double nextDisplayTime;

};

PV::BaseObject * createSoundProbe(char const * name, PV::HyPerCol * hc);

#endif // SOUNDPROBE_HPP_
