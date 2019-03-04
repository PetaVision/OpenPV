/*
 * CorrectState.hpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#ifndef CORRECTSTATE_HPP_
#define CORRECTSTATE_HPP_

#include "probes/ColProbe.hpp"

#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

/**
 * A class used by HyPerConnCheckpointerTest to encapsulate the correct values
 * for the input and output layers and for the weight between them, and the
 * number of updates.
 *
 * The update rule is:
 *
 * input increments by 1 at timesteps 5, 9, 13, etc.
 * new output = new input * old weight.
 * new weight = old weight + new input * new output, updates at t=4, 8, 12, etc.
 *
 * The test uses an input layer display of 4 and a connection that, either
 * using the input layer as a trigger layer, with triggerOffset=1, or
 * by defining weightUpdatePeriod=4 and initialWeightUpdateTime=4,
 * updates at times 4, 8, 12, 16, and 20.
 *
 * The frames of the input layer are constant 1, constant 2, etc, and the
 * weight has dWMax = 1. The weight is initialized as 1 and the output layer
 * is initialized as 2. Therefore, at the end of each timestep, the correct
 * state of the system is as follows:
 *
 *   time update  input   output  connection
 *     0     0      1         1        1
 *     1     1      1         1        1
 *     2     1      1         1        1
 *     3     1      1         1        1
 *     4     1      1         1        2
 *     5     2      2         4        2
 *     6     2      2         4        2
 *     7     2      2         4        2
 *     8     2      2         4       10
 *     9     3      3        30       10
 *    10     3      3        30       10
 *    11     3      3        30       10
 *    12     3      3        30      100
 *    13     4      4       400      100
 *    14     4      4       400      100
 *    15     4      4       400      100
 *    16     4      4       400     1700
 *    17     5      5      8500     1700
 *    18     5      5      8500     1700
 *    19     5      5      8500     1700
 *    20     5      5      8500    44200
 *
 */
class CorrectState {
  public:
   /**
    * Public constructor for the CorrectState class, setting the initial update number,
    * weight value, input value, and output value.
    */
   CorrectState(
         int initialUpdateNumber,
         float initialWeight,
         float initialInput,
         float initialOutput);

   /**
    * Destructor for the CorrectState class.
    */
   virtual ~CorrectState() {}

   /**
    * Applies the update rule to get the next weight, input and output values,
    * and increments the update number.
    */
   void update();

   /**
    * Returns the current update number.
    */
   int getTimestamp() const { return mTimestamp; }

   /**
    * Returns the current value for the correct weight.
    */
   float getCorrectWeight() const { return mCorrectWeight; }

   /**
    * Returns the current value for the correct input.
    */
   float getCorrectInput() const { return mCorrectInput; }

   /**
    * Returns the current value for the correct output.
    */
   float getCorrectOutput() const { return mCorrectOutput; }

  private:
   int mTimestamp       = 0;
   float mCorrectWeight = 0.0;
   float mCorrectInput  = 0.0;
   float mCorrectOutput = 0.0;
};

#endif // CORRECTSTATE_HPP_
