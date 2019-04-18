/*
 * CorrectState.cpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#include "CorrectState.hpp"
#include <climits>
#include <cmath>

CorrectState::CorrectState(
      int initialUpdateNumber,
      PVLayerLoc const *inputLoc,
      PVLayerLoc const *outputLoc)
      : mUpdateNumber(initialUpdateNumber), mInputLoc(*inputLoc) {
   mCorrectInputBuffer.resize(inputLoc->nx, inputLoc->ny, inputLoc->nf);
   mCorrectOutputBuffer.resize(outputLoc->nx, outputLoc->ny, outputLoc->nf);
}

void CorrectState::update() {
   mUpdateNumber++;
   updateCorrectInputBuffer();
   updateCorrectOutputBuffer();
}

void CorrectState::updateCorrectInputBuffer() {
   int const nx        = mCorrectInputBuffer.getWidth();
   int const ny        = mCorrectInputBuffer.getHeight();
   int const nf        = mCorrectInputBuffer.getFeatures();
   int const N         = mCorrectInputBuffer.getTotalElements();
   int const numGlobal = mInputLoc.nxGlobal * mInputLoc.nyGlobal * mInputLoc.nf;
   for (int n = 0; n < N; n++) {
      int const x = kxPos(n, nx, ny, nf);
      int const y = kyPos(n, nx, ny, nf);
      int const f = featureIndex(n, nx, ny, nf);
      int nGlobal = globalIndexFromLocal(n, mInputLoc);
      mCorrectInputBuffer.set(x, y, f, (float)((nGlobal + mUpdateNumber) % numGlobal));
   }
}

void CorrectState::updateCorrectOutputBuffer() {
   pvAssert(mCorrectInputBuffer.getWidth() % mCorrectOutputBuffer.getWidth() == 0);
   int const cellSizeX = mCorrectInputBuffer.getWidth() / mCorrectOutputBuffer.getWidth();
   pvAssert(mCorrectInputBuffer.getHeight() % mCorrectOutputBuffer.getHeight() == 0);
   int const cellSizeY = mCorrectInputBuffer.getHeight() / mCorrectOutputBuffer.getHeight();

   int const nx = mCorrectOutputBuffer.getWidth();
   int const ny = mCorrectOutputBuffer.getHeight();
   int const nf = mCorrectOutputBuffer.getFeatures();
   int const N  = mCorrectOutputBuffer.getTotalElements();
   for (int n = 0; n < N; n++) {
      int const x = kxPos(n, nx, ny, nf);
      int const y = kyPos(n, nx, ny, nf);
      int const f = featureIndex(n, nx, ny, nf);

      float currentMax = -FLT_MAX;
      for (int ycell = 0; ycell < cellSizeY; ycell++) {
         for (int xcell = 0; xcell < cellSizeX; xcell++) {
            float input = mCorrectInputBuffer.at(x * cellSizeX + xcell, y * cellSizeY + ycell, f);
            currentMax  = std::max(currentMax, input);
         }
      }
      mCorrectOutputBuffer.set(x, y, f, currentMax);
   }
}
