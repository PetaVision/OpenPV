/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  A subclass of BaseInput that processes an image based on an existing memory
 *  buffer instead of reading from a file.
 *
 *  Before using the image (typically after initializing the object but before
 *  calling the parent HyPerCol's run method), call the setMemoryBuffer() method.
 *  If using buildandrun, setMemoryBuffer() can be called using the custominit hook.
 */

#ifndef IMAGEFROMMEMORYBUFFER_HPP_
#define IMAGEFROMMEMORYBUFFER_HPP_

#include "BaseInput.hpp"

namespace PV {

class ImageFromMemoryBuffer : public BaseInput{

public:
   ImageFromMemoryBuffer(char const * name, HyPerCol * hc);
   
   virtual ~ImageFromMemoryBuffer();
   
   /**
    * Sets the image.  Under MPI, nonroot processes ignore the externalBuffer
    * argument; all other arguments must be the same across all processes.
    *
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

   /**
    * Returns the factor by which the image was resized when setMemoryBuffer is called:
    * If autoResizeFlag is true, this factor is the larger of (layer's nx/memory buffer's nx) and (layer's ny/memory buffer's ny).
    * If autoResizeFlag is false, this factor is always 1.
    */
   inline float getResizeFactor() const { return resizeFactor; }
   inline int getImageLeft() const { return imageLeft; }
   inline int getImageRight() const { return imageRight; }
   inline int getImageTop() const { return imageTop; }
   inline int getImageBottom() const { return imageBottom; }

protected:
   ImageFromMemoryBuffer();
   
   int initialize(char const * name, HyPerCol * hc);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /** 
    * List of parameters needed from the ImageFromMemoryBuffer class
    * @name Image Parameters
    * @{
    */

   /**
    * @brief inputPath: Not used by ImageFromMemoryBuffer.
    * @details ImageFromMemoryBuffer does not read the input from a path.  Instead, call setMemoryBuffer()
    */
   virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag) { return; }
   
   /**
    * @brief autoResizeFlag: resize image before cropping to the layer
    * @details If set to true, image will be resized to the
    * smallest size with the same aspect ratio that completely covers the
    * layer size, and then cropped according to the offsets and offsetAnchor
    * parameters inherited from BaseInput.
    */
   virtual void ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief aspectRatioAdjustment: either "crop" or "pad"
    * @details If autoResizeFlag is true * and the input buffer's aspect ratio
    * is different from the layer's, this parameter controls whether to
    * resize the image so that it completely covers the layer and then crop;
    * or to resize the image to completely fit inside the layer and then pad.
    */
   virtual void ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag);

   /**
    * Called by HyPerLayer::setActivity() during setInitialValues stage; calls copyBuffer()
    */
   virtual int initializeActivity(double time, double dt);
   
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

   /**
    * Calculates the dimensions in local extended (n.b. check this) coordinates for the given rank.
    * of the region occupied by the layer.
    * Each process calls this method with its own rank to calculate imageLeft, imageRight, etc.
    * The root process calls it for all ranks to determine what part of the image to scatter to the other processes.
    */
   int calcLocalBox(int rank, int * dataLeft, int * dataTop, int * imageLeft, int * imageTop, int * width, int * height);

   int retrieveData(double timef, double dt);
      
private:
   int initialize_base();
   
   template <typename pixeltype> pvadata_t pixelTypeConvert(pixeltype q, pixeltype zeroval, pixeltype oneval);

   /**
    * Used by setMemoryBuffer when autoResizeFlag is set.
    * Performs band-by-band bicubic interpolation of the input buffer,
    * placing the result in the buffer member variable.
    * Inputs:
    *    bufferIn    A pointer to the beffer containing the image.
    *                Under MPI, only the root process uses buffer and the root process scatters the image to the other processes.
    *    heightIn    The height in pixels of the entire image
    *    widthIn     The width in pixels of the entire image
    *    numbands    The number of bands in the image: i.e., grayscale=1, RGB=3, etc.
    *    xStrideIn   The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the x-direction, with the same y-coordinate and band number.
    *    yStrideIn   The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the y-direction, with the same x-coordinate and band number.
    *    bandStrideIn The difference between the memory locations, as pointers of type pixeltype, between two pixels from adjacent bands, with the same x- and y-coordinates.
    *    zeroval     The value that should be converted to 0.0f internally.
    *    oneval      The value that should be converted to 1.0f internally.  Values other than zeroval and oneval are converted to floats using a linear transformation.
    */
   template <typename pixeltype> int bicubicinterp(pixeltype const * bufferIn, int heightIn, int widthIn, int numBands, int xStrideIn, int yStrideIn, int bandStdrideIn, int heightOut, int widthOut, pixeltype zeroval, pixeltype oneval);

   // Bicubic convolution kernel with a=-1
   inline static pvadata_t bicubic(pvadata_t x) {
      pvadata_t const absx = fabsf(x); // assumes pvadata_t is float ; ideally should generalize
      return absx < 1 ? 1 + absx*absx*(-2 + absx) : absx < 2 ? 4 + absx*(-8 + absx*(5-absx)) : 0;

   }


// Member variables
protected:
   pvadata_t * buffer;
   int bufferSize;
   bool hasNewImageFlag; // set to true by setMemoryBuffer; cleared to false by initializeActivity();
   bool autoResizeFlag;
   char * aspectRatioAdjustment;
   float resizeFactor;
   int imageLeft; // image{Left,Right,Top,Bottom} are in local layer coordinates.
   int imageRight;// They show what part of the local layer is occupied by the image.
   int imageTop;
   int imageBottom;
}; // class ImageFromMemoryBuffer

BaseObject * createImageFromMemoryBuffer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif // IMAGEFROMMEMORYBUFFER_HPP_
