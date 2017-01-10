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
 * A class used by MomentumConnViscosityCheckpointerTest to encapsulate the
 * correct values for the input and output layers and for the weight between
 * them, and the number of updates.
 *
 * This test uses momentumMethod="viscosity". The value of momentumTau,
 * 1/log(2)=1.4426950408889634,
 * was chosen so that the tauFactor=exp(-1/momentumTau) = 0.5 is an
 * easy-to-work-with value.
 *
 * The update rule for momentumMethod=viscosity, momentumDecay=0, dWMax=1, and
 * momentumTau = 1/log(2)=1.4426950408889634 is:
 *
 * new update number = old update number + 1.
 * new dw = old input * old output + 0.5*old dw
 * new weight = old weight + new dw.
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
 *   time update  dw  weight input   output
 *     0     0      0     1    1         2
 *     1     1      2     3    1         3
 *     2     1      2     3    1         3
 *     3     1      2     3    1         3
 *     4     1      2     3    1         3
 *     5     2      4     7    2        14
 *     6     2      4     7    2        14
 *     7     2      4     7    2        14
 *     8     2      4     7    2        14
 *     9     3     30    37    3       111
 *    10     3     30    37    3       111
 *    11     3     30    37    3       111
 *    12     3     30    37    3       111
 *    13     4    348   385    4      1540
 *    14     4    348   385    4      1540
 *    15     4    348   385    4      1540
 *    16     4    348   385    4      1540
 *    17     5   6334  6719    5     33595
 *    18     5   6334  6719    5     33595
 *    19     5   6334  6719    5     33595
 *    20     5   6334  6719    5     33595
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
