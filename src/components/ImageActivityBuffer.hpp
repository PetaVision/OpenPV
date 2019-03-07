/*
 * ImageActivityBuffer.hpp
 *
 *  Created on: Jul 22, 2015
 *      Author: Sheng Lundquist
 */

#ifndef IMAGEACTIVITYUPDATER_HPP_
#define IMAGEACTIVITYUPDATER_HPP_

#include "components/InputActivityBuffer.hpp"
#include "structures/Image.hpp"

namespace PV {

/**
 * A component for the activity buffer for ImageLayer
 */
class ImageActivityBuffer : public InputActivityBuffer {
  public:
   ImageActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ImageActivityBuffer();

   virtual std::string const &
   getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const override;

  protected:
   ImageActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   /**
    * ImageActivityBuffer does not register any additional data structures for checkpointing.
    * If the InputPath is a URL, the template for temporary filenames for file downloads is
    * set here, to a path in the OutputPath directory.
    */
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * If the InputPath ends in .txt, returns the length of the list of filenames.
    * If it is an input image, returns 1.
    * to give the BatchIndexer the number of input images.
    */
   virtual int countInputImages() override;

   /**
    * Fills the FileList with either the filenames appearing in InputPath if it is a list of files,
    * or the InputPath filename if it is a single image.
    */
   void populateFileList();

   /**
    * Returns the filename corresponding to the indicated (zero-indexed) index.
    */
   virtual std::string describeInput(int index) override;

   /**
    * Reads the file indicated by the inputIndex argument into the mImage data member.
    * inputIndex is the (zero-indexed) index into the list of inputs.
    */
   virtual Buffer<float> retrieveData(int inputIndex) override;

   virtual void readImage(std::string filename);

  protected:
   std::unique_ptr<Image> mImage = nullptr;

   // Automatically set if the inputPath ends in .txt. Determines whether this layer represents a
   // collection of files.
   bool mUsingFileList = false;

   // List of filenames to iterate over
   std::vector<std::string> mFileList;

   // Template for a temporary path for downloading URLs that appear in file list.
   std::string mURLDownloadTemplate;
};

} // namespace PV

#endif // IMAGEACTIVITYUPDATER_HPP_
