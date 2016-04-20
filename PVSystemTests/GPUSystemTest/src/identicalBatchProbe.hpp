/*
 * identicalBatchProbe.hpp
 * Author: slundquist
 */

#ifndef IDENTICALFEATUREPROBE_HPP_ 
#define IDENTICALFEATUREPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class identicalBatchProbe : public PV::StatsProbe{
public:
   identicalBatchProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initidenticalBatchProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initidenticalBatchProbe_base();

};

BaseObject * create_identicalBatchProbe(char const * probeName, HyPerCol * hc);

}
#endif 
