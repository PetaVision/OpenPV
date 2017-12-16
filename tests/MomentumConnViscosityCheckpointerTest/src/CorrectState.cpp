/*
 * CorrectState.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "CorrectState.hpp"

CorrectState::CorrectState(
      int initialUpdateNumber,
      float initialWeight,
      float initial_dw,
      float initialInput,
      float initialOutput)
      : mUpdateNumber(initialUpdateNumber),
        mCorrectWeight(initialWeight),
        mCorrect_dw(initial_dw),
        mCorrectInput(initialInput),
        mCorrectOutput(initialOutput) {}

void CorrectState::update() {
   mUpdateNumber++;
   mCorrect_dw = 0.5f * mCorrect_dw + (mCorrectInput * mCorrectOutput);
   mCorrectWeight += mCorrect_dw;
   mCorrectInput  = (float)mUpdateNumber;
   mCorrectOutput = mCorrectInput * mCorrectWeight;
}
