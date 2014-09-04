/*
 *  Timer.hpp
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
   class Timer {
   public:
      Timer(double init_time=0.0);
      Timer(const char * timermessage, double init_time=0.0);
      Timer(const char * objname, const char * objtype, const char * timertype, double init_time=0.0);
      virtual ~Timer();
      void reset(double init_time=0.0);

      virtual double start();
      virtual double stop();
      inline double elapsed_time();
      virtual int fprint_time(FILE * stream);
      
   protected:
      
      int rank;
      char * message;
      
      uint64_t time_start, time_end;
      uint64_t time_elapsed;

   };
   
} // namespace PV

#endif /* TIMER_HPP_ */
