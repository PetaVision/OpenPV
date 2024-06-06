/*
 * PvpListActivityBuffer.hpp
 *
 *  Created on: Aug 31, 2022
 *      Author: Pete Schultz
 */

#ifndef IMAGEACTIVITYUPDATER_HPP_
#define IMAGEACTIVITYUPDATER_HPP_

#include "components/InputActivityBuffer.hpp"
#include "structures/Image.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

/**
 * A component for the activity buffer for PvpListLayer
 */
class PvpListActivityBuffer : public InputActivityBuffer {
  public:
   PvpListActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PvpListActivityBuffer();

   virtual std::string const &
   getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const override;

  protected:
   PvpListActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   /**
    * returns the length of the list of filenames,
    * to give the BatchIndexer the number of input images.
    */
   virtual int countInputImages() override;

   /**
    * Fills the FileList with the filenames appearing in the InputPath file
    */
   void populateFileList();

   /**
    * Returns the filename corresponding to the indicated (zero-indexed) index.
    */
   virtual std::string describeInput(int index) override;

   /**
    * Reads the file indicated by the inputIndex argument into a buffer
    * inputIndex is the (zero-indexed) index into the list of inputs.
    */
   virtual Buffer<float> retrieveData(int inputIndex) override;

  protected:
   // List of filenames to iterate over
   std::vector<std::string> mFileList;
};

} // namespace PV

#endif // IMAGEACTIVITYUPDATER_HPP_
