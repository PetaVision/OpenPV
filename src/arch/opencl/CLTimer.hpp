/*
 * CLTimer.hpp
 *
 * Sheng Lundquist
 */

#ifndef CLTIMER_HPP_
#define CLTIMER_HPP_

#include "pv_opencl.h"
#include "../../utils/Timer.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace PV {

class CLTimer : public Timer{
#ifdef PV_USE_OPENCL
   public:
      CLTimer(double init_time=0.0);
      CLTimer(const char * timermessage, double init_time=0.0);
      CLTimer(const char * objname, const char * objtype, const char * timertype, double init_time=0.0);
      virtual ~CLTimer();

      double accumulateTime();
      virtual int fprint_time(FILE * stream);
      cl_event * getTimerEvent(){return timerEvent;}
      
   private:
      cl_event * timerEvent;
      float time;
#endif
};

}
#endif
