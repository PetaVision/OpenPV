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
 * A class used by MomentumConnSimpleCheckpointerTest to encapsulate the correct values
 * for the input and output layers and for the weight between them, and the
 * number of updates.
 *
 * The update rule for momentumMethod=simple, momentumTau=1, momentumDecay=0,
 * dWMax=1 is:
 *
 * new update number = old update number + 1.
 * new dw = old dw + (old input * old output).
 * new weight = old weight + dw.
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
 *   time update weight  dw  input   output
 *     0     0       1     0   1         2
 *     1     1       3     2   1         3
 *     2     1       3     2   1         3
 *     3     1       3     2   1         3
 *     4     1       3     2   1         3
 *     5     2       8     5   2        16
 *     6     2       8     5   2        16
 *     7     2       8     5   2        16
 *     8     2       8     5   2        16
 *     9     3      45    37   3       135
 *    10     3      45    37   3       135
 *    11     3      45    37   3       135
 *    12     3      45    37   3       135
 *    13     4     487   442   4      1948
 *    14     4     487   442   4      1948
 *    15     4     487   442   4      1948
 *    16     4     487   442   4      1948
 *    17     5    8721  8234   5     43605
 *    18     5    8721  8234   5     43605
 *    19     5    8721  8234   5     43605
 *    20     5    8721  8234   5     43605
 *
 */
class CorrectState {
  public:
   /**
    * Public constructor for the CorrectState class, setting the initial update number,
    * weight value, input value, and output value.
    */
   CorrectState(
         int intialUpdateNumber,
         float initialWeight,
         float initial_dw,
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
    * Returns the current value for the correct dw.
    */
   float getCorrect_dw() const { return mCorrect_dw; }

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
   float mCorrect_dw    = 0.0;
   float mCorrectInput  = 0.0;
   float mCorrectOutput = 0.0;
};

#endif // CORRECTSTATE_HPP_
