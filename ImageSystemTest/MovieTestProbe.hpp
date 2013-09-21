/*
 * MovieTestProbe.hpp
 * Author: slundquist
 */

#ifndef MOVIETESTPROBE_HPP_
#define MOVIETESTPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class MovieTestProbe : public PV::StatsProbe{
public:
   MovieTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   MovieTestProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
};

}
#endif /* IMAGETESTPROBE_HPP */
