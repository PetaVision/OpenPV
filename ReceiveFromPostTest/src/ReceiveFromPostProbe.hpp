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
   ReceiveFromPostProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initReceiveFromPostProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initReceiveFromPostProbe_base();

};

}
#endif 
