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
      float initialInput,
      float initialOutput)
      : mUpdateNumber(initialUpdateNumber),
        mCorrectWeight(initialWeight),
        mCorrectInput(initialInput),
        mCorrectOutput(initialOutput) {}

void CorrectState::update() {
   mUpdateNumber++;
   mCorrectWeight += mCorrectInput * mCorrectOutput;
   mCorrectInput  = (float)mUpdateNumber;
   mCorrectOutput = mCorrectInput * mCorrectWeight;
}
