/*
 * CorrectState.hpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#ifndef CORRECTSTATE_HPP_
#define CORRECTSTATE_HPP_

#include "probes/ColProbe.hpp"

#include "structures/Buffer.hpp"

/**
 * A class used by PoolingConnCheckpointerTest to encapsulate the correct values
 * for the input and output layers and for the weight between them, and the
 * number of updates.
 *
 * The test uses a 4x4 input layer display of 4. Therefore, the the first update
 * happens at t=1, update 2 at t=5, update 3 at t=9, etc. Update n occurs at
 * time 4*n-3.
 *
 * The output layer is a 2x2 HyPerLayer, and the PoolingConn uses maxpooling has
 * with nxp,nyp = 1. Therefore each 2x2 nonoverlapping tile of the input layer
 * is pooled into the output neuron above its center.
 *
 * On update n, the input layer is
 *     n      n+1      n+2      n+3
 *     n+4    n+5      n+6      n+7
 *     n+8    n+9      n+10     n+11
 *     n+12   n+13     n+14     n+15
 * where the addition is modulo 16. For example, at t=13, i.e. update 4,
 * the input layer is
 *     4      5        6        7
 *     8      9       10       11
 *    12     13       14       15
 *     0      1        2        3
 *
 * and the output layer is
 *        9                11
 *
 *       13                15
 */
class CorrectState {
  public:
   /**
    * Public constructor for the CorrectState class, setting the initial update number,
    * weight value, input value, and output value.
    */
   CorrectState(int initialUpdateNumber, PVLayerLoc const *inputLoc, PVLayerLoc const *outputLoc);

   /**
    * Destructor for the CorrectState class.
    */
   virtual ~CorrectState() {}

   /**
    * Applies the update rule to get the next set of correct input and output
    * values, and increments the update number.
    */
   void update();

   /**
    * Returns the current update number.
    */
   int getUpdateNumber() const { return mUpdateNumber; }

   /**
    * Returns a reference to the current buffer for the correct input.
    */
   PV::Buffer<float> const &getCorrectInputBuffer() const { return mCorrectInputBuffer; }

   /**
    * Returns a reference to the current buffer for the correct output.
    */
   PV::Buffer<float> const &getCorrectOutputBuffer() const { return mCorrectOutputBuffer; }

  private:
   /**
    * Called by update() after incrementing the update number.
    * Uses the update number to compute what the input buffer should be
    * (see documentation of CorrectState class).
    */
   void updateCorrectInputBuffer();

   /**
    * Called by update() after calling updateCorrectInputBuffer().
    * Computes what the output buffer should be, from pooling the correct input buffer.
    */
   void updateCorrectOutputBuffer();

  private:
   int mUpdateNumber = 0;
   PV::Buffer<float> mCorrectInputBuffer;
   PVLayerLoc const mInputLoc;
   PV::Buffer<float> mCorrectOutputBuffer;
};

#endif // CORRECTSTATE_HPP_
