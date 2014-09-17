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
   MovieTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initMovieTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initMovieTestProbe_base();
};

}
#endif /* IMAGETESTPROBE_HPP */
