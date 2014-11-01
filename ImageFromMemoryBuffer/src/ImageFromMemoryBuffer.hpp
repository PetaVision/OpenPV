/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  A subclass of Image that processes an image based on an existing memory
 *  buffer instead of reading from a file.
 *
 *  Before using the image (typically after initializing the object but before
 *  calling the parent HyPerCol's run method), call the setMemoryBuffer() method.
 *  If using buildandrun, setMemoryBuffer() can be called using the custominit hook.
 */

#ifndef IMAGEFROMMEMORYBUFFER_HPP_
#define IMAGEFROMMEMORYBUFFER_HPP_

#include <layers/Image.hpp>

namespace PV {

class ImageFromMemoryBuffer : public Image {

public:
   ImageFromMemoryBuffer(char const * name, HyPerCol * hc);
   
   virtual ~ImageFromMemoryBuffer();
   
   /**
    * Sets the image.
    * Inputs:
    *    buffer      A pointer to the beffer containing the image.
    *                Under MPI, only the root process uses buffer and the root process scatters the image to the other processes.
    *    height      The height in pixels of the entire image
    *    width       The width in pixels of the entire image
    *    numbands    The number of bands in the image: i.e., grayscale=1, RGB=3, etc.
    *    xstride     The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the x-direction, with the same y-coordinate and band number.
    *    ystride     The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the y-direction, with the same x-coordinate and band number.
    *    bandstride  The difference between the memory locations, as pointers of type pixeltype, between two pixels from adjacent bands, with the same x- and y-coordinates.
    *    zeroval     The value that should be converted to 0.0f internally.
    *    oneval      The value that should be converted to 1.0f internally.  Values other than zeroval and oneval are converted to floats using a linear transformation.
    */
   template <typename pixeltype> int setMemoryBuffer(pixeltype const * externalBuffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, pixeltype zeroval, pixeltype oneval);

protected:
   ImageFromMemoryBuffer();
   
   int initialize(char const * name, HyPerCol * hc);
   
   /** 
    * List of parameters needed from the ImageFromMemoryBuffer class
    * @name Image Parameters
    * @{
    */

   /**
    * @brief imagePath: Not used by ImageFromMemoryBuffer.
    * @details ImageFromMemoryBuffer does not read the image from a path.  Instead, call setMemoryBuffer()
    */
   virtual void ioParam_imagePath(enum ParamsIOFlag ioFlag) { return; }
   
   /**
    * Copies data from the buffer and scatters it to the several MPI processes
    */
   virtual int initializeActivity();
   
   /**
    * Under MPI, this function may only be called by the rank-zero process.
    * Finds the portion of the buffer that corresponds to the process whose rank is the input argument,
    * and copies it into the data buffer.  It does not call any MPI sends; the calling routine
    * needs to do so.   (This is the common code for sending to nonroot and root processes)
    * 
    */
   int moveBufferToData(int rank);
   
private:
   int initialize_base();
   
   template <typename pixeltype> pvadata_t pixelTypeConvert(pixeltype q, pixeltype zeroval, pixeltype oneval);

// Member variables
protected:
   pvadata_t * buffer;
}; // class ImageFromMemoryBuffer

}  // namespace PV

#endif // IMAGEFROMMEMORYBUFFER_HPP_
