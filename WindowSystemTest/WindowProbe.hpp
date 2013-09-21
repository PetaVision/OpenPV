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
   WindowProbe(const char * filename, HyPerLayer * layer, const char * msg);
   WindowProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
};

}
#endif /* WINDOWPROBE_HPP */
