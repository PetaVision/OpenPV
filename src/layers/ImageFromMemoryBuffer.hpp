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

#include "Image.hpp"

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

   /**
    * Sets the image.  Overloads setMemoryBuffer to also change the parameters offsetX, offsetY, and offsetAnchor.
    */
   template <typename pixeltype> int setMemoryBuffer(pixeltype const * externalBuffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, pixeltype zeroval, pixeltype oneval, int offsetX, int offsetY, char const * offsetAnchor);

   /**
    * Returns true if a new image has been set by a call to setMemoryBuffer without having been copied to the
    * activity buffer by a call to copyBuffer() (which is called by either updateState or initializeActivity)
    */
   virtual bool needUpdate(double time, double dt) { return hasNewImageFlag; }
   
   /**
    * For ImageFromMemoryBuffer, the updateTime is the parent->getStopTime() - parent->getStartTime().
    * Implemented to allow triggering off of an ImageFromMemoryBuffer layer.
    */
   virtual double getDeltaUpdateTime();

/**
    * Overrides updateState
    */
   virtual int updateState(double time, double dt);

   /**
    * ImageFromMemoryBuffer uses the same outputState as HyPerLayer
    */
   virtual int outputState(double time, bool last=false);

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
    * Called by HyPerLayer::setActivity() during setInitialValues stage; calls copyBuffer()
    */
   virtual int initializeActivity();
   
   /**
    * Copies the contents of the image buffer to the activity buffer.
    * Under MPI, the image buffer is scattered to the several processes.
    */
   int copyBuffer();
   
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
   bool hasNewImageFlag; // set to true by setMemoryBuffer; cleared to false by initializeActivity();
}; // class ImageFromMemoryBuffer

}  // namespace PV

#endif // IMAGEFROMMEMORYBUFFER_HPP_
