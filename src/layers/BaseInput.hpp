/**
 * Base Input class for reading images, movies, pvp files, as well as creating patterns and loading from memory buffer.
 */
#ifndef BASEINPUT_HPP_
#define BASEINPUT_HPP_

#include "HyPerLayer.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/Random.hpp"
#include "io/imageio.hpp"
#include "utils/Buffer.hpp"

#include <memory>

namespace PV {

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
         virtual int ioParam_offsets(enum ParamsIOFlag ioFlag); 
         
         /**
          * @brief offsetAnchor: Defines where the anchor point is for the offsets.
          * @details Specified as a 2 character string, "xy" <br />
          * x can be 'l', 'c', or 'r' for left, center, right respectively <br />
          * y can be 't', 'c', or 'b' for top, center, bottom respectively <br />
          * Defaults to "tl"
          */
         virtual void ioParam_offsetAnchor(enum ParamsIOFlag ioFlag);

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
          * @brief padValue: If the image is being padded (image smaller than layer), the value to use for padding
          */
         virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

        /**
          * @brief offsetConstraintMethod: If jitter flag is set, defines the method to coerce into bounding box
          * @details Can be 0 (ignore), 1 (mirror BC), 2 (threshold), or 3 (circular BC)
          */
         virtual void ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag);
         /**
          * @brief initVType: Image does not have a V, do not set
          */
         virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

         /**
          * @brief triggerLayerName: BaseInput and derived classes do not use triggering, and always set triggerLayerName to NULL.
          */
         virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

          /**
          * @brief displayPeriod: the amount of time each image is displayed before switching to the next image.
          * The units of displayPeriod are the same as the units of the HyPerCol's dt parameter.
          */
         virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

         /**
          * @brief echoFramePathnameFlag: if true, print the filename to the screen when a new image file is loaded.
          */
         virtual void ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag);

         /**
          * @brief start_frame_index: Initialize the layer with the given frame.
          * @details start_frame_index=0 means the first line of the imageListPath if a text file,
          * or the initial frame if imageListPath is a .pvp file.
          */
         virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);

         /**
          * @brief skip_frame_index: If skip_frame_index=1, go to the next frame at the end of the display period.
          * If skip_frame_index=2, skip the next frame and go to the second frame after the frame that just expired,
          * and so on.  If skip_frame_index is less than one, it behaves the same as skip_frame_index=1.
          */
         virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);
         /**
          * @brief writeFrameToTimestamp: if true, then every time the frame is updated, it writes the frame number, the time and the image filename
          *  to a file.  The file is placed in a directory "timestamps" in the outputPath directory, and the filename is the layer name appended with ".txt".
          */
         virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);

         /**
          * @brief flipOnTimescaleError: determines whether to change images at the end of the display period if the HyPerCol's timescale is less than the HyPerCol's timeScaleMin (defaults to true)
          */
         virtual void ioParam_flipOnTimescaleError(enum ParamsIOFlag ioFlag);

         /**
          * @brief resetToStartOnLoop: If false, then when the end of file for the imageListPath file is reached, it rewinds to the beginning of the file.
          * If true, it resets to the location given by start_frame_index.
          */
         virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);

         /**
          * @brief batchMethod: Specifies how to split the file for batches.
          * byImage: Each batch skips nbatch, and starts staggered from the same part of the file 
          * byMovie: Each batch skips 1, and starts at numFrames/numBatch part of the file 
          * bySpecified: User specified start_frame_index and skip_frame_index, one for each batch
          */
         virtual void ioParam_batchMethod(enum ParamsIOFlag ioFlag);
         /**
          * @brief useImageBCFlag: Specifies if the Image layer should use the image to fill margins 
          */
         virtual void ioParam_useInputBCflag(enum ParamsIOFlag ioFlag);
         /**
          * @}
          */

      protected:
         enum BatchMethod {
            BYFILE,
            BYLIST,
            BYSPECIFIED
         };

         BaseInput();
         int initialize(const char * name, HyPerCol * hc);
         virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

         int initRandState();

         virtual int allocateV();
         virtual int initializeV();
         virtual int initializeActivity();
         static bool constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method);
         virtual bool constrainOffsets();
         virtual int readStateFromCheckpoint(const char *cpDir, double *timeptr);
                  /**
          * This is the interface for loading a new "frame" (which can be either pvp, image, etc)
          * into the activity buffer. This function calls retrieveData, scatterInput, and postProcess.
          */
         void nextInput(double timef, double dt);

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
         int calcLocalBox(int rank, int &dataLeft, int &dataTop, int &imageLeft, int &imageTop, int &width, int &height);

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


      public:
         BaseInput(const char * name, HyPerCol * hc);
         virtual ~BaseInput();
         virtual int requireChannel(int channelNeeded, int * numChannelsResult);
         virtual int allocateDataStructures();
         virtual int updateState(double time, double dt);
         virtual int checkpointRead(const char * cpDir, double * timeptr);
         virtual int checkpointWrite(const char *cpDir);
         virtual double calcTimeScale(int batchIndex);

         virtual bool activityIsSpiking() {return false;}
         void exchange();

         int getOffsetX(const char* offsetAnchor, int offsetX);
         int getOffsetY(const char* offsetAnchor, int offsetY);
         /**
          * getImageStartX() returns the x-coordinate in the original input corresponding to x=0 in layer coordinates.
          */
         int getImageStartX() { return getOffsetX(mOffsetAnchor, mOffsets[0]); }

         /**
          * getImageStartY() returns the y-coordinate in the original input corresponding to y=0 in the layer coordinates.
          */
         int getImageStartY() { return getOffsetX(mOffsetAnchor, mOffsets[0]); }
         
         // These get-methods are needed for masking
         int getDataLeft() { return mLayerLeft; }
         int getDataTop() { return mLayerTop; }
         int getImageLeft() { return mInputLeft; }
         int getImageTop() { return mInputTop; }
         int getDataWidth() { return mInputWidth; }
         int getDataHeight() { return mInputHeight; }

      private:
         int initialize_base();
         int populateFileList();
         std::string getNextFilename(int filesToSkip, int batchIndex);
         std::string advanceFilename(int batchIndex);

      protected:
         MPI_Datatype * mDatatypes;  // MPI datatypes for boundary exchange

         Buffer mInputData; //Raw data read from disk. 
         bool mAutoResizeFlag;
         char * mAspectRatioAdjustment;
         Buffer::InterpolationMethod mInterpolationMethod;
         float mResizeFactor;
         bool mInverseFlag;
         bool mNormalizeLuminanceFlag; // if true, normalize the input image as specified by normalizeStdDev
         bool mNormalizeStdDev;        // if true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
                                      // if false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
                                      //
         int mOffsetConstraintMethod; // If offsets escape the bounding box, the method to coerce them into the bounding box.
                                // The constraint method codes are 0=ignore, 1=mirror boundary conditions, 2=thresholding, 3=circular boundary conditions
         int mOffsets[2];        // offsets array points to [offsetX, offsetY]
         char* mOffsetAnchor;

         int mLayerLeft; // The left edge of valid image data in the local activity buffer.  Can be positive if there is padding.  Can be negative if the data extends into the border region.
         int mLayerTop; // The top edge of valid image data in the local activity buffer.  Can be positive if there is padding.  Can be negative if the data extends into the border region.
         int mInputLeft; // The x-coordinate in image coordinates corresponding to a value of dataLeft in layer coordinates.
         int mInputTop; // The y-coordinate in image coordinates corresponding to a value of dataTop in layer coordinates.
         int mInputWidth; // The width of valid image data in local activity buffer.
         int mInputHeight; // The height of valid image data in the local activity buffer.
         float mPadValue;
         bool mUseInputBCflag;

         std::string mInputPath;

         PV_Stream *mTimestampFile;

      private:
         double mDisplayPeriod;   // length of time a frame is displayed

         bool mEchoFramePathnameFlag; // if true, echo the frame pathname to output stream
         bool mResetToStartOnLoop;

         std::vector<int> mStartFrameIndex;
         std::vector<int> mSkipFrameIndex;
         std::string mInputFile;  // current input file name

         int mNumFiles; //Number of frames
         std::vector<std::string> mFilePaths;
         BatchMethod mBatchMethod;

         PV_Stream *mFilenameStream;
         
         std::vector<int> mFileNumbers; // Index inside the list of files that each batch is at

         bool mWriteFileToTimestamp;

         bool mFlipOnTimescaleError;

   }; // class BaseInput
}  // namespace PV

#endif // BASEINPUT_HPP_

