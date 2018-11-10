/*
 * PvpActivityBuffer.hpp
 *
 *  Created on: Aug 16, 2016
 *      Author: Austin Thresher
 */

#ifndef PVPACTIVITYBUFFER_HPP_
#define PVPACTIVITYBUFFER_HPP_

#include "components/InputActivityBuffer.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

/**
 * A component for the activity buffer for PvpLayer
 */
class PvpActivityBuffer : public InputActivityBuffer {
  public:
   PvpActivityBuffer(char const *name, HyPerCol *hc);

   virtual ~PvpActivityBuffer();

  protected:
   PvpActivityBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   /**
    * Returns the number of frames in the InputPath pvp file
    */
   virtual int countInputImages() override;

   /**
    * Reads the file indicated by the inputIndex argument into the mImage data member.
    * inputIndex is the (zero-indexed) index into the list of inputs.
    */
   virtual Buffer<float> retrieveData(int inputIndex) override;

  private:
   struct BufferUtils::SparseFileTable mSparseTable;
};

} // namespace PV

#endif // PVPACTIVITYBUFFER_HPP_
