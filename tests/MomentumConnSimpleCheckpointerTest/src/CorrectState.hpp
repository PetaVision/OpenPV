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
 * The update rule for momentumMethod=simple, momentumDecay=0, dWMax=1, and
 * timeConstantTau = 0.75, is:
 *
 * input starts at 1 and increments by 1 at timesteps 5, 9, 13, etc.
 * new output = new input * old weight.
 * Weight and DeltaWeight update at timesteps 4, 8, 12, etc. using the rule
 * new dw = 0.75*old dw + 0.25*(dWMax * new input * new output).
 * new weight = old weight + new dw.
 *
 * The test uses an input layer displayPeriod of 4 and a connection that, either
 * using the input layer as a trigger layer or by defining weightUpdatePeriod=4
 * and initialWeightUpdateTime=1, updates at times 1, 5, 9, etc.
 *
 * The frames of the input layer are constant 1, constant 2, etc, and the
 * weight has dWMax = 1. The weight is initialized as 1 and the output layer
 * is initialized as 2. Therefore, at the end of each timestep, the correct
 * state of the system is as follows:
 *
 *  time       input      output      pre*post      dw        weight
 *   0           1           1           1          0           1
 *   1           1           1           1          0           1
 *   2           1           1           1          0           1
 *   3           1           1           1          0           1
 *   4           1           1           1          0.25        1.25
 *   5           2           2.5         5          0.25        1.25
 *   6           2           2.5         5          0.25        1.25
 *   7           2           2.5         5          0.25        1.25
 *   8           2           2.5         5          1.4375      2.6875
 *   9           3           8.0625      24.1875    1.4375      2.6875
 *  10           3           8.0625      24.1875    1.4375      2.6875
 *  11           3           8.0625      24.1875    1.4375      2.6875
 *  12           3           8.0625      24.1875    7.125       9.8125
 *  13           4          39.25       157         7.125       9.8125
 *  14           4          39.25       157         7.125       9.8125
 *  15           4          39.25       157         7.125       9.8125
 *  16           4          39.25       157        44.59375    54.40625
 *  17           5         272.03125   1360.15625  44.59375    54.40625
 *  18           5         272.03125   1360.15625  44.59375    54.40625
 *  19           5         272.03125   1360.15625  44.59375    54.40625
 *  20           5         272.03125   1360.15625 373.484375  427.890625
 */
class CorrectState {
  public:
   /**
    * Public constructor for the CorrectState class, setting the initial update number,
    * weight value, input value, and output value.
    */
   CorrectState(
         float timeConstantTau,
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
    * Returns true if weights should update during the specified timestep, false otherwise.
    */
   bool doesWeightUpdate(double timestamp) const;

   /**
    * Returns the time constant tau.
    */
   float getTimeConstantTau() const { return mTimeConstantTau; }

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

   /**
    * Returns the number of times update() has been called.
    */
   float getTimestamp() const { return mTimestamp; }

  private:
   float mTimeConstantTau = 0.0f;
   float mCorrectWeight   = 0.0f;
   float mCorrect_dw      = 0.0f;
   float mCorrectInput    = 0.0f;
   float mCorrectOutput   = 0.0f;

   int mTimestamp = 0;
};

#endif // CORRECTSTATE_HPP_
