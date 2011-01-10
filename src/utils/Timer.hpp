/*
 *  Timer.hpp
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
   class Timer {
   public:
      Timer();
      void reset();

      virtual double start();
      virtual double stop();
      virtual double elapsed_time();
      
   protected:
      
      int rank;
      
      uint64_t time_start, time_end;
      uint64_t time_elapsed;

   };
   
} // namespace PV

#endif /* TIMER_HPP_ */
