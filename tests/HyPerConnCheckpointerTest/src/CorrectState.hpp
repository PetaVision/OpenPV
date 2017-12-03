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
 * new update number = old update number + 1.
 * new weight = old weight + old input + old output.
 * new input  = new update number.
 * new output = new input * new weight.
 *
 * The test uses an input layer display of 4 and a connection that, either
 * using the input layer as a trigger layer or by defining weightUpdatePeriod=4
 * and initialWeightUpdateTime=1, updates at times 1, 5, 9, etc.
 *
 * The frames of the input layer are constant 1, constant 2, etc, and the
 * weight has dWMax = 1. The weight is initialized as 1 and the output layer
 * is initialized as 2. Therefore, at the end of each timestep, the correct
 * state of the system is as follows:
 *
 *   time update connection  input   output
 *     0     0         1       1         2
 *     1     1         3       1         3
 *     2     1         3       1         3
 *     3     1         3       1         3
 *     4     1         3       1         3
 *     5     2         6       2        12
 *     6     2         6       2        12
 *     7     2         6       2        12
 *     8     2         6       2        12
 *     9     3        30       3        90
 *    10     3        30       3        90
 *    11     3        30       3        90
 *    12     3        30       3        90
 *    13     4       300       4      1200
 *    14     4       300       4      1200
 *    15     4       300       4      1200
 *    16     4       300       4      1200
 *    17     5      5100       5     25500
 *    18     5      5100       5     25500
 *    19     5      5100       5     25500
 *    20     5      5100       5     25500
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
   int getUpdateNumber() const { return mUpdateNumber; }

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
   int mUpdateNumber    = 0;
   float mCorrectWeight = 0.0;
   float mCorrectInput  = 0.0;
   float mCorrectOutput = 0.0;
};

#endif // CORRECTSTATE_HPP_
