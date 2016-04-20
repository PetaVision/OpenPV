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
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);
   void ioParam_tolerance(enum ParamsIOFlag ioFlag);

private:
   int initReceiveFromPostProbe_base();

// Member variables
protected:
   pvadata_t tolerance;

}; // end class ReceiveFromPostProbe

BaseObject * createReceiveFromPostProbe(char const * name, HyPerCol * hc);

}  // end namespcae PV

#endif  // RECEIVEFROMPOSTPROBE_HPP_
