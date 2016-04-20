/**
 * Base Input class for reading images, movies, pvp files, as well as creating patterns and loading from memory buffer.
 */
#ifndef BASEINPUT_HPP_
#define BASEINPUT_HPP_

#include "HyPerLayer.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/Random.hpp"
#include "io/imageio.hpp"


namespace PV {

typedef enum {
   INTERPOLATE_UNDEFINED,
   INTERPOLATE_BICUBIC,
   INTERPOLATE_NEARESTNEIGHBOR
} InputInterpolationMethod;

typedef enum {
   COLORTYPE_UNRECOGNIZED,
   COLORTYPE_GRAYSCALE/*One or two bands; if two the second is the alpha channel*/,
   COLORTYPE_RGB/*Three or four bands; if four, the last is the alpha channel*/
} InputColorType;

class BaseInput: public HyPerLayer{
protected:
   
   /** 
    * List of parameters needed from the BaseInput class
    * @name BaseInput Parameters
    * @{
    */

   /**
    * @brief inputPath: The file the input is reading. The type of file depends on which subclass is being used.
    */
   virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag);

   /**
    * @brief offsetX: offset in X direction <br />
    * offsetY: offset in Y direction
    * @details Defines an offset in image space where the column is viewing the image
    */
   virtual int ioParam_offsets(enum ParamsIOFlag ioFlag); // reads offsetX, offsetY from params.  Override with empty function if a derived class doesn't use these parameters (e.g. Patterns)

   /**
    * @brief offsetAnchor: Defines where the anchor point is for the offsets.
    * @details Specified as a 2 character string, "xy" <br />
    * x can be 'l', 'c', or 'r' for left, center, right respectively <br />
    * y can be 't', 'c', or 'b' for top, center, bottom respectively <br />
    * Defaults to "tl"
    */
   virtual void ioParam_offsetAnchor(enum ParamsIOFlag ioFlag);

   /** 
    * @brief writeImages: A boolean flag that specifies if the Image class should output the images.
    */
   virtual void ioParam_writeImages(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeImageExtension: If writeImages is set, specifies the extention of the image output.
    * @details Defaults to .tif
    */
   virtual void ioParam_writeImagesExtension(enum ParamsIOFlag ioFlag);

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
    * @brief interpolationMethod: either "bicubic" or "nearestNeighbor".
    */
   virtual void ioParam_interpolationMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief inverseFlag: If set to true, inverts the image
    */
   virtual void ioParam_inverseFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief normalizeLuminanceFlag: If set to true, will normalize the image.
    * The normalization method is determined by the normalizeStdDev parameter.
    */
   virtual void ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief normalizeStdDev: This flag is used if normalizeLuminanceFlag is true.
    * If normalizeStdDev is set to true, the image will normalize with a mean of 0 and std of 1
    * If normalizeStdDev is set to false, the image will normalize with a min of 0 and a max of 1
    */
   virtual void ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterFlag: If set to true, will move the image around by specified pixels
    */
   virtual void ioParam_jitterFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterType: If jitter flag is set, specifies the type of jitter. 0 for random walk, 1 for random jump
    * @details - Random Walk: Jitters the specified step in any direction
    * - Random Jump: Jitters any value between -step and step in any direction
    */
   virtual void ioParam_jitterType(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterRefactoryPeriod: If jitter flag is set, specifies the minimum amount of time until next jitter
    */
   virtual void ioParam_jitterRefractoryPeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief stepSize: If jitter flag is set, sets the step size
    */
   virtual void ioParam_stepSize(enum ParamsIOFlag ioFlag);

   /**
    * @brief persistenceProb: If jitter flag is set, sets the probability that offset stays the same
    */
   virtual void ioParam_persistenceProb(enum ParamsIOFlag ioFlag);

   /**
    * @brief recurrenceProb: If jitter flag is set, sets the probability that the offset returns to bias position
    */
   virtual void ioParam_recurrenceProb(enum ParamsIOFlag ioFlag);

   /**
    * @brief padValue: If the image is being padded (image smaller than layer), the value to use for padding
    */
   virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

   /**
    * @brief biasChangeTime: If jitter flag is set, sets the time period for recalculating bias position
    */
   virtual void ioParam_biasChangeTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief biasConstraintMethod: If jitter flag is set, defines the method to coerce into bounding box
    * @details Can be 0 (ignore), 1 (mirror BC), 2 (threshold), or 3 (circular BC)
    */
   virtual void ioParam_biasConstraintMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief offsetConstraintMethod: If jitter flag is set, defines the method to coerce into bounding box
    * @details Can be 0 (ignore), 1 (mirror BC), 2 (threshold), or 3 (circular BC)
    */
   virtual void ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief writePosition: If jitter flag is set, writes position to input/image-pos.txt
    */
   virtual void ioParam_writePosition(enum ParamsIOFlag ioFlag);

   /**
    * @brief initVType: Image does not have a V, do not set
    */
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: BaseInput and derived classes do not use triggering, and always set triggerLayerName to NULL.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   //TODO this functionality should be in both pvp and image. Set here for now, as pvp does not support imageBC
   /**
    * @brief useImageBCFlag: Specifies if the Image layer should use the image to fill margins 
    */
   virtual void ioParam_useImageBCflag(enum ParamsIOFlag ioFlag);

   /**
    * @}
    */

protected:
   BaseInput();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int initRandState();

   virtual int allocateV();
   virtual int initializeV();
   virtual int initializeActivity();

   virtual bool jitter();
   virtual int calcBias(int current_bias, int step, int sizeLength);
   virtual int calcNewBiases(int stepSize);
   virtual int calcBiasedOffset(int bias, int current_offset, int step, int sizeLength);
   virtual bool calcNewOffsets(int stepSize);
   static bool constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method);
   virtual bool constrainBiases();
   virtual bool constrainOffsets();

   /**
    * This is the interface for loading a new "frame" (which can be either pvp, image, etc)
    * into the activity buffer. This function calls retrieveData, scatterInput, and postProcess.
    */
   virtual int getFrame(double timef, double dt);

   /**
    * This pure virtual function gets called from getFrame by the root process only.
    * Derived classes should set the nxGlobal, nyGlobal, and nf fields of imageLoc.
    * If the product of nxGlobal, nyGlobal, and nf changes, retrieveData should
    * free imageData with delete[] and reallocate imageData with new[], to prevent
    * memory leaks.  retrieveData should also set the imageColorType data member.
    */
   virtual int retrieveData(double timef, double dt, int batchIndex) = 0;

   /**
    * This function scatters the imageData buffer to the activity buffers of the several MPI processes.
    */
   virtual int scatterInput(int batchIndex);

   /**
    * Calculates the intersection of the given rank's local extended region
    * with the imageData, based on the offsetX, offsetY, and offsetAnchor
    * parameters.
    * Used in scatterInput by the root process to determine what part of the
    * imageData buffer to scatter to the other processes.
    * Return value is zero if width and height are both positive, and nonzero
    * if either is negative (i.e. the local layer and image do not intersect).
    */
   int calcLocalBox(int rank, int * dataLeft, int * dataTop, int * imageLeft, int * imageTop, int * width, int * height);

   /**
    * This function achieves post processing of the activity buffer after a frame is loaded.
    */
   virtual int postProcess(double timef, double dt);

   /**
    * Returns PV_SUCCESS if offsetAnchor is a valid anchor string
    * (two characters long; first characters one of 't', 'c', or 'b'; second characters one of 'l', 'c', or 'r')
    * Returns PV_FAILURE otherwise
    */
   int checkValidAnchorString();

   int copyFromInteriorBuffer(float * buf, int batchIdx, float fac);
   int copyToInteriorBuffer(unsigned char * buf, int batchIdx, float fac);

   /**
    * This method is called during scatterInput, by the root process only.
    * It uses the values of autoResizeFlag, aspectRatioAdjustment, and ImageBCflag
    * to resize the imageData buffer.  It also updates the nxGlobal, nyGlobal, and nf
    * fields of imageLoc, and the resizeFactor data member.
    */
   virtual int resizeInput();

   int nearestNeighborInterp(pvadata_t const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStdrideIn, pvadata_t * bufferOut, int widthOut, int heightOut);

   /**
    * Resizes an image using band-by-band bicubic interpolation of the bufferIn array,
    * placing the result in the bufferOut array.
    * bufferIn is widthIn-by-heightOut-by-numBands; bufferOut is widthOut-by-heightOut-by-numBands.
    * Inputs:
    *    bufferIn    A pointer to the buffer containing the image.
    *    widthIn     The width in pixels of the entire image
    *    heightIn    The height in pixels of the entire image
    *    numbands    The number of bands in the image: i.e., grayscale=1, RGB=3, etc.
    *    xStrideIn   The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the x-direction, with the same y-coordinate and band number.
    *    yStrideIn   The difference between the memory locations, as pointers of type pixeltype, between two pixels adjacent in the y-direction, with the same x-coordinate and band number.
    *    bandStrideIn The difference between the memory locations, as pointers of type pixeltype, between two pixels from adjacent bands, with the same x- and y-coordinates.
    *    bufferOut   The buffer for the resized image
    *    widthOut    The width in pixels of the entire image
    *    heightOut   The height in pixels of the entire image
    */
   int bicubicInterp(pvadata_t const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStdrideIn, pvadata_t * bufferOut, int widthOut, int heightOut);

   // Bicubic convolution kernel with a=-1
   inline static pvadata_t bicubic(pvadata_t x) {
      pvadata_t const absx = fabsf(x); // assumes pvadata_t is float ; ideally should generalize
      return absx < 1 ? 1 + absx*absx*(-2 + absx) : absx < 2 ? 4 + absx*(-8 + absx*(5-absx)) : 0;

   }

   /**
    * Converts a grayscale buffer to a multiband buffer, by replicating the buffer in each band.
    * On entry, *buffer points to an nx-by-ny-by-1 buffer that must have been created with the new[] operator.
    * On exit, *buffer points to an nx-by-ny-by-numBands buffer that was created with the new[] operator.
    */
   static int convertGrayScaleToMultiBand(float ** buffer, int nx, int ny, int numBands);

   /**
    * Converts a multiband buffer to a grayscale buffer, using the colorType to weight the bands.
    * On entry, *buffer points to an nx-by-ny-by-numBands buffer that must have been created with the new[] operator.
    * On exit, *buffer points to an nx-by-ny-by-1 buffer that was created with the new[] operator.
    */
   static int convertToGrayScale(float ** buffer, int nx, int ny, int numBands, InputColorType colorType);

   /**
    * Based on the value of colorType, fills the bandweights buffer with weights to assign to each band
    * of a multiband buffer when converting to grayscale.
    */
   static int calcBandWeights(int numBands, float * bandweights, InputColorType colorType);

   /**
    * Called by calcBandWeights when the color type is unrecognized; it fills each bandweights entry
    * with 1/numBands.
    */
   static inline void equalBandWeights(int numBands, float * bandweights) {
      float w = 1.0/(float) numBands;
      for( int b=0; b<numBands; b++ ) bandweights[b] = w;
   }

public:
   BaseInput(const char * name, HyPerCol * hc);
   virtual ~BaseInput();
   virtual int requireChannel(int channelNeeded, int * numChannelsResult);
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);

   virtual int checkpointRead(const char * cpDir, double * timeptr);

   virtual bool activityIsSpiking() {return false;}

   virtual PVLayerLoc getImageLoc() {return imageLoc; }

   int writeImage(const char * filename, int batchIdx);

   int exchange();

   int getOffsetX(const char* offsetAnchor, int offsetX);
   int getOffsetY(const char* offsetAnchor, int offsetY);

   /**
    * getImageStartX() returns the x-coordinate in the original input corresponding to x=0 in layer coordinates.
    */
   int getImageStartX() { return getOffsetX(offsetAnchor, offsets[0]); }

   /**
    * getImageStartY() returns the y-coordinate in the original input corresponding to y=0 in the layer coordinates.
    */
   int getImageStartY() { return getOffsetX(offsetAnchor, offsets[0]); }
   
   int getBiasX() { return biases[0]; }
   int getBiasY() { return biases[1]; }
   const int * getBiases() { return biases; }
   char * getInputPath(){return inputPath;}

   // These get-methods are needed for masking
   int getDataLeft() { return dataLeft; }
   int getDataTop() { return dataTop; }
   int getImageLeft() { return imageLeft; }
   int getImageTop() { return imageTop; }
   int getDataWidth() { return dataWidth; }
   int getDataHeight() { return dataHeight; }

private:
   int initialize_base();

protected:
   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   pvdata_t * data;       // buffer containing reduced image

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image, used only by a local communicator's root process.
   InputColorType imageColorType; // Whether the data in imageData is grayscale, RGB, or something else.

   int writeImages;      // controls writing of image file during outputState
   char * writeImagesExtension; // ".pvp", ".tif", ".png", etc.; the extension to use when writing images
   
   bool autoResizeFlag;
   char * aspectRatioAdjustment;
   InputInterpolationMethod interpolationMethod;
   float resizeFactor;
   bool inverseFlag;
   bool normalizeLuminanceFlag; // if true, normalize the input image as specified by normalizeStdDev
   bool normalizeStdDev;        // if true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
                                // if false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
                                //
   // Jitter parameters
   int jitterFlag;        // If true, use jitter
   int stepSize;
   int biases[2];         // When jittering, the distribution for the offset is centered on offsetX=biasX, offsetY=biasY
                          // Jittering can change the bias point on a slower timescale than the offset point changes.
   int biasConstraintMethod;  // If biases escape the bounding box, the method to coerce them into the bounding box.
   int offsetConstraintMethod; // If offsets escape the bounding box, the method to coerce them into the bounding box.
                          // The constraint method codes are 0=ignore, 1=mirror boundary conditions, 2=thresholding, 3=circular boundary conditions
   float recurrenceProb;  // If using jitter, probability that offset returns to bias position
   float persistenceProb; // If using jitter, probability that offset stays the same
   int writePosition;     // If using jitter, write positions to input/image-pos.txt
   PV_Stream * fp_pos;    // If writePosition is true, write the positions to this file
   double biasChangeTime;    // If using jitter, time period for recalculating bias position
   double nextBiasChange;    // The next time biasChange will be called
   int jitterRefractoryPeriod; // After jitter, minimum amount of time until next jitter
   int timeSinceLastJitter; // Keeps track of timesteps since last jitter
   int jitterType;       // If using jitter, specify type of jitter (random walk or random jump)
   const static int RANDOM_WALK = 0;  // const denoting jitter is a random walk
   const static int RANDOM_JUMP = 1;  // const denoting jitter is a random jump

   int offsets[2];        // offsets array points to [offsetX, offsetY]
   char* offsetAnchor;

   int dataLeft; // The left edge of valid image data in the local activity buffer.  Can be positive if there is padding.  Can be negative if the data extends into the border region.
   int dataTop; // The top edge of valid image data in the local activity buffer.  Can be positive if there is padding.  Can be negative if the data extends into the border region.
   int imageLeft; // The x-coordinate in image coordinates corresponding to a value of dataLeft in layer coordinates.
   int imageTop; // The y-coordinate in image coordinates corresponding to a value of dataTop in layer coordinates.
   int dataWidth; // The width of valid image data in local activity buffer.
   int dataHeight; // The height of valid image data in the local activity buffer.

   float padValue;
   bool useImageBCflag;

   Random * randState;

   char* inputPath;

}; // class BaseInput
}  // namespace PV

#endif // BASEINPUT_HPP_

