/*
 * CorrectState.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "CorrectState.hpp"
#include <cmath>

CorrectState::CorrectState(
      int initialTimestamp,
      float initialWeight,
      float initialInput,
      float initialOutput)
      : mTimestamp(initialTimestamp),
        mCorrectWeight(initialWeight),
        mCorrectInput(initialInput),
        mCorrectOutput(initialOutput) {}

void CorrectState::update() {
   mTimestamp++;
   mCorrectInput  = std::ceil((float)mTimestamp / 4.0f);
   mCorrectOutput = mCorrectInput * mCorrectWeight;
   if (std::fabs(std::fmod(mTimestamp + 1, 4.0) - 1) < 0.5) {
      mCorrectWeight += mCorrectInput * mCorrectOutput;
   }
}
