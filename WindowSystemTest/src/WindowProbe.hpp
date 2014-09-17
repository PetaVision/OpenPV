/*
 * WindowProbe.hpp
 * Author: slundquist
 */

#ifndef WINDOWPROBE_HPP_
#define WINDOWPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV{

class WindowProbe : public PV::StatsProbe{
public:
   WindowProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initWindowProbe(const char * probeName, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initWindowProbe_base();
};

}
#endif /* WINDOWPROBE_HPP */
