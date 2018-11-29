/*
 * CorrectState.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "CorrectState.hpp"

CorrectState::CorrectState(
      float timeConstantTau,
      float initialWeight,
      float initial_dw,
      float initialInput,
      float initialOutput)
      : mTimeConstantTau(timeConstantTau),
        mCorrectWeight(initialWeight),
        mCorrect_dw(initial_dw),
        mCorrectInput(initialInput),
        mCorrectOutput(initialOutput) {}

void CorrectState::update() {
   mUpdateNumber++;
   auto base_dw = mCorrectInput * mCorrectOutput;
   auto prev_dw = mCorrect_dw;
   mCorrect_dw  = (1 - mTimeConstantTau) * base_dw + mTimeConstantTau * prev_dw;
   mCorrectWeight += mCorrect_dw;
   mCorrectInput  = (float)mUpdateNumber;
   mCorrectOutput = mCorrectInput * mCorrectWeight;
}
