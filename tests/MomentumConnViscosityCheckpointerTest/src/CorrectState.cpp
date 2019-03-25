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
   mTimestamp++;
   mCorrectInput  = std::ceil((float)mTimestamp / 4.0f);
   mCorrectOutput = mCorrectInput * mCorrectWeight;
   if (doesWeightUpdate(mTimestamp)) {
      auto base_dw = mCorrectInput * mCorrectOutput;
      mCorrect_dw  = (1 - mTimeConstantTau) * base_dw + mTimeConstantTau * mCorrect_dw;
      mCorrectWeight += mCorrect_dw;
   }
}

bool CorrectState::doesWeightUpdate(double timevalue) const {
   return std::fabs(std::fmod(timevalue + 1, 4.0) - 1) < 0.5;
}
