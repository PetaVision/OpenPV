/*
 * ReceiveFromPostProbe.hpp
 * Author: slundquist
 */

#ifndef RECEIVEFROMPOSTPROBE_HPP_
#define RECEIVEFROMPOSTPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class ReceiveFromPostProbe : public PV::StatsProbe{
public:
   ReceiveFromPostProbe(const char * filename, HyPerLayer * layer, const char * msg);
   ReceiveFromPostProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
};

}
#endif 
