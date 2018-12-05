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
 * This test uses momentumMethod="viscosity". The value of timeConstantTau,
 * -1/log(0.75)=3.4760594967822072,
 * was chosen so that the tauFactor=exp(-1/timeConstantTau) = 0.75 is an
 * easy-to-work-with value.
 *
 * The update rule for momentumMethod=viscosity, momentumDecay=0, dWMax=1, and
 * exp(-1/timeConstantTau) = 0.75 is:
 *
 * new update number = old update number + 1.
 * new dw = 0.75*old dw + 0.25*(dWMax * old input * old output).
 * new weight = old weight + new dw.
 * new input  = new update number.
 * new output = new input * new weight.
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
 *   time        update  pre*post   prev_dw     dw        weight      input       output
 *     0           0                             0         1           1           2
 *     1           1        2        0           0.5       1.5         1           1.5
 *     5           2        1.5      0.5         0.75      2.25        2           4.5
 *     9           3        9.0      0.75        2.8125    5.0625      3          15.1875
 *    13           4       45.5625   2.8125     13.5      18.5625      4          74.25
 *    17           5      297       13.5        84.375   102.9375      5         514.6875
 *    20           5      297       13.5        84.375   102.9375      5         514.6875
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
   float getUpdateNumber() const { return mUpdateNumber; }

  private:
   float mTimeConstantTau = 0.0;
   float mCorrectWeight   = 0.0;
   float mCorrect_dw      = 0.0;
   float mCorrectInput    = 0.0;
   float mCorrectOutput   = 0.0;

   int mUpdateNumber = 0;
};

#endif // CORRECTSTATE_HPP_
