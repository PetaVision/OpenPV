/*
 * GPUSystemTestProbe.hpp
 * Author: slundquist
 */

#ifndef GPUSYSTEMTESTPROBE_HPP_ 
#define GPUSYSTEMTESTPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class GPUSystemTestProbe : public PV::StatsProbe{
public:
   GPUSystemTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initGPUSystemTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initGPUSystemTestProbe_base();

};

BaseObject * createGPUSystemTestProbe(char const * probeName, HyPerCol * hc);

}
#endif 
