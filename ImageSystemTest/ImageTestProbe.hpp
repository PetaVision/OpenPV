/*
 * ImageTestProbe.hpp
 * Author: slundquist
 */

#ifndef IMAGETESTPROBE_HPP_
#define IMAGETESTPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class ImageTestProbe : public PV::StatsProbe{
public:
   ImageTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   ImageTestProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
};

}
#endif /* IMAGETESTPROBE_HPP */
