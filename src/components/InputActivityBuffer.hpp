/*
 * InputActivityBuffer.hpp
 *
 *  Created on: Jul 22, 2015
 *      Author: Sheng Lundquist
 */

#ifndef INPUTACTIVITYBUFFER_HPP_
#define INPUTACTIVITYBUFFER_HPP_

#include "checkpointing/CheckpointableFileStream.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/BatchIndexer.hpp"
#include "utils/BufferUtilsRescale.hpp"
#include <random>

namespace PV {

/**
 * A component for the activity updater for BinningLayer.
 */
class InputActivityBuffer : public ActivityBuffer {
  protected:
   /**
    * List of parameters used by the InputActivityBuffer class
    * @name InputLayer Parameters
    * @{
    */

   /**
    * inputPath: Either an individual file to load, or a .txt list of files to load.
    */
   virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag);

   /**
    * offsetX: offset in X direction
    * offsetY: offset in Y direction
    * Defines an offset in image space where the column is viewing the image
    */
   virtual void ioParam_offsets(enum ParamsIOFlag ioFlag);

   /**
    * jitterChangeInterval:
    * Defines the period of the random shifts updates
    * A value less than or equal to zero means no jitter.
    */
   virtual void ioParam_jitterChangeInterval(enum ParamsIOFlag ioFlag);

   /**
    * jitterChangeIntervalUnit:
    * Either "displayPeriod" or "timestep" (case-insensitive).
    * Whether jitterChangeInterval is the number of display periods or
    * the number of timesteps.
    * Defaults to displayPeriod.
    */
   virtual void ioParam_jitterChangeIntervalUnit(enum ParamsIOFlag ioFlag);

   /**
    * maxShiftX: max random shift in X direction
    * maxShiftY: max random shift in Y direction
    * Defines the max random shift in image space
    */
   virtual void ioParam_maxShifts(enum ParamsIOFlag ioFlag);

   /**
    * xFlipEnabled: When true, 50% chance to mirror input horizontally
    * yFlipEnabled: When true, 50% chance to mirror input vertically
    */
   virtual void ioParam_flipsEnabled(enum ParamsIOFlag ioFlag);

   /**
    * xFlipToggle: When true, flip every jitter interval instead of randomly
    * yFlipToggle: When true, flip every jitter interval instead of randomly
    */
   virtual void ioParam_flipsToggle(enum ParamsIOFlag ioFlag);

   /**
    * offsetAnchor: Defines where the anchor point is for the offsets.
    * Specified as a 2 character string, "xy"
    * x can be 'l', 'c', or 'r' for left, center, right respectively <br />
    * y can be 't', 'c', or 'b' for top, center, bottom respectively <br />
    */
   virtual void ioParam_offsetAnchor(enum ParamsIOFlag ioFlag);

   /**
    * autoResizeFlag: Whether to scale the image to fit layer dimensions
    */
   virtual void ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag);

   /**
    * aspectRatioAdjustment: either "crop" or "pad"
    */
   virtual void ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag);

   /**
    * interpolationMethod: either "bicubic" or "nearestNeighbor".
    */
   virtual void ioParam_interpolationMethod(enum ParamsIOFlag ioFlag);

   /**
    * inverseFlag: If set to true, inverts the input: pixels are mapped linearly
    * so that the max pixel value is mapped to the min and vice versa.
    */
   virtual void ioParam_inverseFlag(enum ParamsIOFlag ioFlag);

   /**
    * normalizeLuminanceFlag: If set to true, will normalize the image.
    * The normalization method is determined by the normalizeStdDev parameter.
    */
   virtual void ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag);

   /**
    * normalizeStdDev: This flag is used if normalizeLuminanceFlag is true.
    * If normalizeStdDev is set to true, the image will normalize with a mean of 0 and std of 1
    * If normalizeStdDev is set to false, the image will normalize with a min of 0 and a max of 1
    * If all pixels are equal, the image will normalize so that all pixels are zero.
    */
   virtual void ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag);

   /**
    * padValue: If the image is being padded (image smaller than layer), the value to use for
    * padding
    */
   virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

   /**
    * displayPeriod: the number of timesteps each input is displayed before switching to the next.
    * If this is <= 0 or inputPath does not end in .txt, assumes the input is a single file and will
    * not change.
    */
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

   /**
    * start_frame_index: Array specifying the file indices to start at.
    * If displayPeriod <= 0, this determines which index from the file list will be used.
    */
   virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);

   /**
    * skip_frame_index: Array specifying how much to increment the file index by each displayPeriod
    * for each batch
    */
   virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);

   /**
    * writeFrameToTimestamp: if true, then every time the frame is updated, it writes the frame
    * number,
    * the time and the image filename to a file. The file is placed in a directory "timestamps" in
    * the outputPath
    * directory, and the filename is the layer name appended with ".txt".
    */
   virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);

   /**
    * resetToStartOnLoop: If false, then when the end of file for the inputPath file is reached,
    * it rewinds to index 0. Otherwise, it rewinds to the index it began at (possibly
    * start_frame_index).
    */
   virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);

   /**
    * batchMethod: Specifies how to split the file for batches.
    * byFile: Each batch skips nbatch, and starts staggered from the beginning of the file list
    * byList: Each batch skips 1, and starts at index = numFrames/numBatch
    * bySpecified: User specified start_frame_index and skip_frame_index, one for each batch
    * random: Randomizes the order of the given file. Does not duplicate indices until all are used
    */
   virtual void ioParam_batchMethod(enum ParamsIOFlag ioFlag);

   /**
    * Random seed used when batchMethod == random.
    */
   virtual void ioParam_randomSeed(enum ParamsIOFlag ioFlag);

   /**
    * useInputBCFlag: Specifies if the input should be scaled to fill margins
    */
   virtual void ioParam_useInputBCflag(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   InputActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~InputActivityBuffer();

   void makeInputRegionsPointer(ActivityBuffer *activityBuffer);

   virtual std::string const &getCurrentFilename(int localBatchIndex, int mpiBatchIndex) const;

   float const *getInputRegionsAllBatchElements() const {
      return mInputRegionsAllBatchElements.data();
   }

   std::string const &getInputPath() const { return mInputPath; }

   int getDisplayPeriod() const { return mDisplayPeriod; }

  protected:
   InputActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * Returns PV_SUCCESS if offsetAnchor is a valid anchor string, PV_FAILURE otherwise.
    * (two characters long; first characters one of 't', 'c', or 'b'; second characters one of 'l',
    * 'c', or 'r')
    */
   int checkValidAnchorString(const char *offsetAnchor);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   void initializeBatchIndexer();

   /**
    * This virtual function gets called by initializeBatchIndexer in order
    * to give the BatchIndexer the number of input images.
    */
   virtual int countInputImages() { return 0; }

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   /**
    * Each batch element loads its data with the input specified by the current
    * index for that batch element. The indices are not modified.
    * Called both to initialize the layer and to flip the image during updateActivity.
    */
   void retrieveInput(double timef, double dt);

   void writeToTimestampStream(double simTime);

   /**
    * This function is called by writeToTimestampStream if the writeFrameToTimestamp flag is set.
    * It provides the opportunity for a subclass to insert a description of the input, along with
    * the timestamp and the name of the InputLayer. The input argument is the index into the list
    * of input frames. The default is to return the empty string. However, ImageActivityUpdater
    * overrides this to return the finelname corresponding to the input index.
    */
   virtual std::string describeInput(int index) { return std::string(""); }

   /**
    * Each batch element loads its input by calling retrieveInput(), and then advances its index
    * by the amount specified by skip_frame_index. Called by updateActivity().
    */
   void retrieveInputAndAdvanceIndex(double timef, double dt);

   /**
    * Method that signals when to load the next file. Returns true if displayPeriod is positive,
    * negative otherwise. Can be overridden for different file list logic in derived classes.
    */
   virtual bool readyForNextFile(double simTime, double deltaT);

   /**
    * This virtual function gets called by the root process during retrieveInput,
    * int both the InitializeState stage and during updateState. It loads the entire input
    * (scattering to nonroot processes is done by the scatterInput method)
    * into a buffer. inputIndex is the (zero-indexed) index into the list of inputs.
    */
   virtual Buffer<float> retrieveData(int inputIndex) { return Buffer<float>(); }

  private:
   /**
    * Resizes a buffer from the image size to the global layer size. If autoResizeFlag is true,
    * it calls BufferUtils::rescale. If autoResizeFlag is false, it calls Buffer methods grow,
    * translate, and crop. This method is called only by the MPI block root process,
    * during retrieveInput().
    */
   void fitBufferToGlobalLayer(Buffer<float> &buffer, int blockBatchElement);

   /**
    * normalizePixels transforms the input on a per-pixel basis specified by the batch element,
    * based on the normalizeLuminanceFlag, normalizeStdDev, and inverseFlag parameters.
    * Overload this to add additional post process steps in subclasses.
    * Pixels not occupied by the actual image (due to offsets, padding, etc.) are not changed.
    * This method is called only by the MPI block root process, during retrieveInput().
    */
   void normalizePixels(int batchElement);
   // Would it allow for more parallelism to move this after the scatterInput() call?

   /**
    * Crops the given buffer, sized to the global layer, to the MPIBlock.
    * This method is called only by the MPI block's root process, during retrieveInput().
    */
   void cropToMPIBlock(Buffer<float> &buffer);

   /**
    * This method scatters the mInputData buffer to the activity buffers of the several MPI
    * processes. It is called by all processes, during retrieveInput().
    */
   void scatterInput(int localBatchIndex, int mpiBatchIndex);

  protected:
   // Number of timesteps an input file is displayed before advancing the file list. If <= 0, the
   // input never changes.
   int mDisplayPeriod = 0;

   // Path to input file or list of input files
   std::string mInputPath;

   // When cropping or resizing, which side of the canvas is the origin?
   Buffer<float>::Anchor mAnchor = Buffer<float>::CENTER;

   // Amount to translate input buffer before scattering but after rescaling
   int mOffsetX = 0;
   int mOffsetY = 0;

   // How often to change random shifts (measured in displayPeriods)
   // Value of zero means no jitter
   int mJitterChangeInterval = 0;

   char *mJitterChangeIntervalUnit = nullptr;

   int mJitterChangeIntervalInTimesteps = 0;

   // If nonzero, create a sample by shifting image randomly in [-maxRandomShiftX, maxRandomShiftX]
   // x [-maxRandomShiftY, maxRandomShiftY]
   int mMaxShiftX = 0;
   int mMaxShiftY = 0;

   // Are horizontal or vertical mirror flips enabled during the augmentation stage?
   bool mXFlipEnabled = false;
   bool mYFlipEnabled = false;

   // If this is true, toggle mirror flips each time instead of randomly selecting
   bool mXFlipToggle = false;
   bool mYFlipToggle = false;

   // Flag that enables rescaling input buffer to layer dimensions instead of just cropping
   bool mAutoResizeFlag = false;

   // If mAutoResizeFlag is enabled, do we crop the edges or pad the edges with mPadValue?
   // Set during read of parameter aspectRatioAdjustment
   BufferUtils::RescaleMethod mRescaleMethod;

   // If mAutoResizeFlag is enabled, do we rescale with bicubic or nearest neighbor filtering?
   BufferUtils::InterpolationMethod mInterpolationMethod = BufferUtils::BICUBIC;

   // Flag that inverts input buffer during post process step
   bool mInverseFlag = false;

   // Flag enabling normalization in the post process step
   bool mNormalizeLuminanceFlag = false;

   // If true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
   // If false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
   bool mNormalizeStdDev = true;

   // Flag that enables scaling input buffer to extended region instead of restricted region
   bool mUseInputBCflag = false;

   // Value to fill empty region with when using padding
   float mPadValue = 0.0f;

   // Object to handle assigning file indices to batch element
   std::unique_ptr<BatchIndexer> mBatchIndexer;
   BatchIndexer::BatchMethod mBatchMethod;

   // Random seed used when batchMethod == random
   int mRandomSeed = 123456789;

   // An array of starting file list indices, one per batch
   std::vector<int> mStartFrameIndex;

   // An array indicating how far to advance each index, one per batch
   std::vector<int> mSkipFrameIndex;

   // When reaching the end of the file list, do we reset to 0 or to start_index?
   // This parameter is read only if using batchMethod=bySpecified
   bool mResetToStartOnLoop = false;

   // Flag to write filenames and batch indices to disk as they are loaded
   bool mWriteFrameToTimestamp = true;

   // Data read from disk, one per batch element.
   std::vector<Buffer<float>> mInputData;

   // The parts of the mInputData buffers occupied by image data, as opposed to gaps created by
   // offsets or resizing with padding. When mInputData[b] is first filled using retrieveData,
   // mInputRegion[b] is created as a buffer of the same size filled with ones.
   // As operations that translate or resize are applied to mInputData[b], the same operation is
   // applied to mInputRegion[b]. When normalizePixels() normalizes or scatterInput() copies to the
   // activity buffer, only those pixels where mInputRegion[b] is nonzero are used.
   std::vector<Buffer<float>> mInputRegion;

   // An array of random shifts in x direction, one per batch
   std::vector<int> mRandomShiftX;
   // An array of random shifts in y direction, one per batch
   std::vector<int> mRandomShiftY;

   // Same for mirror flips
   std::vector<bool> mMirrorFlipX;
   std::vector<bool> mMirrorFlipY;

   // Random number generator for jitter
   std::mt19937 mRNG;

  private:
   bool mNeedInputRegionsPointer = false;

   // A vector containing the contents of the mInputRegion buffers, allocated as a single array
   // of size getNumExtendedAllBatches.
   // Will not be allocated unless the makeInputRegionsPointer() method is called before the
   // AllocateData stage.
   std::vector<float> mInputRegionsAllBatchElements;

   // A vector containing the ActivityBuffer objects that the InputRegion buffers will be
   // copied to.
   std::vector<ActivityBuffer *> mInputRegionTargets;

   // Filepointer to output file used when mWriteFrameToTimestamp == true
   CheckpointableFileStream *mTimestampStream = nullptr;
};

} // namespace PV

#endif // INPUTACTIVITYBUFFER_HPP_
