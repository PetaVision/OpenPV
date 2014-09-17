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
   ImageTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initImageTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initImageTestProbe_base();
};

}
#endif /* IMAGETESTPROBE_HPP */
