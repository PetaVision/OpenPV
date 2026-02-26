/*
 * ImageCollationActivityBuffer.hpp
 */

#ifndef IMAGECOLLATIONACTIVITYBUFFER_HPP_
#define IMAGECOLLATIONACTIVITYBUFFER_HPP_

#include "components/InputActivityBuffer.hpp"
#include "structures/Image.hpp"

namespace PV {

/**
 * A component for the activity buffer for ImageCollationLayer
 */
class ImageCollationActivityBuffer : public InputActivityBuffer {
  public:
   ImageCollationActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ImageCollationActivityBuffer();

   virtual std::string const &
   getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const override;

  protected:
   ImageCollationActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   /**
    * Sets the template for temporary filenames when the image file is a URL.
    * The template is a path in the OutputPath directory, with basename temp.XXXXXX;
    * Files downloaded from a URL are given this basename plus the URL file extension;
    * then XXXXXX is replaced using a call to ::mkstemps().
    */
   virtual Response::Status allocateDataStructures() override;

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
    * Returns the lines within the list of input files corresponding to the indicated
    * (zero-indexed) index.
    */
   virtual std::string describeInput(int index) override;

   /**
    * Reads nf files from the list of inputs at the point indicated by the inputIndex argument,
    * converts each to grayscale, and collates them into a buffer with nf features.
    * The (one-indexed) lines read fom the list of inputs are
    *     (nf * inputIndex + 1) through (nf * (inputIndex + 1)).
    * All the images read during a given call must have the same width and height; it is
    * a fatal error if they do not. (The number of channels can differ since each is converted to
    * grayscale before collation.)
    */
   virtual Buffer<float> retrieveData(int inputIndex) override;

   std::shared_ptr<Image> readImageChannel(std::string const &filename);

   std::string downloadURL(std::string const &url);

  protected:
   std::unique_ptr<Buffer<float>> mImage = nullptr;

   // Automatically set if the inputPath ends in .txt. Determines whether this layer represents a
   // collection of files.
   bool mUsingFileList = false;

   // List of filenames to iterate over
   std::vector<std::string> mFileList;

   // Template for a temporary path for downloading URLs that appear in file list.
   std::string mURLDownloadTemplate;

}; // class ImageCollationActivityBuffer

} // namespace PV

#endif // IMAGECOLLATIONACTIVITYBUFFER_HPP_
