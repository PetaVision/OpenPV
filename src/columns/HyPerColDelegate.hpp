/*
 * HyPerColDelegate.hpp
 * 
 * Virtual base class for HyPerCol runtime delegates.  A
 * runtime delegate replaces the run loop in a HyPerCol.
 * The delegate must call 
 *     HyPerCol::advanceTime(float time);        // within run loop
 *     HyPerCol::exitRunLoop(bool exitOnFinish); // at end of run
 *     
 * A runtime delegate was necessary for OpenGL displays as OpenGL
 * runs its own event loop (never returning).
 *
 *  Created on: Jan 10, 2010
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOLDELEGATE_HPP_
#define HYPERCOLDELEGATE_HPP_

namespace PV {

class HyPerCol;

class HyPerColDelegate {
public:
   HyPerColDelegate();
   virtual ~HyPerColDelegate();

   virtual void run(float time, float stopTime) = 0;
};

}

#endif /* HYPERCOLDELEGATE_HPP_ */
