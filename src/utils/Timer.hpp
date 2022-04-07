/*
 *  Timer.hpp
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <assert.h>
#include <ostream>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/PrintStream.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace PV {

class Timer {
  public:
   Timer(const char *timermessage, double init_time = 0.0);
   Timer(const char *objname, const char *objtype, const char *timertype, double init_time = 0.0);
   virtual ~Timer();
   void reset(double init_time = 0.0);

   virtual double start();
   virtual double stop();
   inline double elapsed_time() const;
   virtual int fprint_time(PrintStream &stream) const;

  protected:
   int rank;
   char *message;

   bool running = false; // start() sets running flag to true; stop() sets it to false.
   uint64_t time_start, time_end;
   uint64_t time_elapsed = (uint64_t)0;
};

} // namespace PV

#endif /* TIMER_HPP_ */
