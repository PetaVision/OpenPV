// InputLayer
// Base class for layers that take their input from file IO

#ifndef __INPUTLAYER_HPP__
#define __INPUTLAYER_HPP__

#include "HyPerLayer.hpp"
#include "checkpointing/CheckpointableFileStream.hpp"
#include "columns/HyPerCol.hpp"
#include "components/BatchIndexer.hpp"
#include "structures/Buffer.hpp"
#include "utils/BorderExchange.hpp"
#include "utils/BufferUtilsRescale.hpp"

#include <memory>
#include <random>

namespace PV {

class InputLayer : public HyPerLayer {
  protected:
   // inputPath: Either an individual file to load, or a .txt list of files to load.
   virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag);

   // offsetX: offset in X direction
   // offsetY: offset in Y direction
   // Defines an offset in image space where the column is viewing the image
   virtual int ioParam_offsets(enum ParamsIOFlag ioFlag);

   // maxShiftX: max random shift in X direction
   // maxShiftY: max random shift in Y direction
   // Defines the max random shift in image space
   virtual int ioParam_maxShifts(enum ParamsIOFlag ioFlag);
   // xFlipEnabled: When true, 50% chance to mirror input horizontally
   // yFlipEnabled: When true, 50% chance to mirror input vertically
   virtual int ioParam_flipsEnabled(enum ParamsIOFlag ioFlag);
   // xFlipToggle: When true, flip every jitter interval instead of randomly
   // yFlipToggle: When true, flip every jitter interval instead of randomly
   virtual int ioParam_flipsToggle(enum ParamsIOFlag ioFlag);
   // jitterChangeInterval: interval measured in displayPeriods
   // Defines the frequency of the random shifts updates
   virtual int ioParam_jitterChangeInterval(enum ParamsIOFlag ioFlag);

   // offsetAnchor: Defines where the anchor point is for the offsets.
   // Specified as a 2 character string, "xy"
   // x can be 'l', 'c', or 'r' for left, center, right respectively <br />
   // y can be 't', 'c', or 'b' for top, center, bottom respectively <br />
   virtual void ioParam_offsetAnchor(enum ParamsIOFlag ioFlag);

   // autoResizeFlag: Whether to scale the image to fit layer dimensions
   virtual void ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag);

   // aspectRatioAdjustment: either "crop" or "pad"
   virtual void ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag);

   // interpolationMethod: either "bicubic" or "nearestNeighbor".
   virtual void ioParam_interpolationMethod(enum ParamsIOFlag ioFlag);

   // inverseFlag: If set to true, inverts the input: pixels are mapped linearly
   // so that the max pixel value is mapped to the min and vice versa.
   virtual void ioParam_inverseFlag(enum ParamsIOFlag ioFlag);

   // normalizeLuminanceFlag: If set to true, will normalize the image.
   // The normalization method is determined by the normalizeStdDev parameter.
   virtual void ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag);

   // normalizeStdDev: This flag is used if normalizeLuminanceFlag is true.
   // If normalizeStdDev is set to true, the image will normalize with a mean of 0 and std of 1
   // If normalizeStdDev is set to false, the image will normalize with a min of 0 and a max of 1
   // If all pixels are equal, the image will normalize so that all pixels are zero.
   virtual void ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag);

   // padValue: If the image is being padded (image smaller than layer), the value to use for
   // padding
   virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

   // initVType: InputLayers do not have a V, do not set
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag) override;

   // triggerLayerName: InputLayer and derived classes do not use triggering, and always set
   // triggerLayerName to NULL.
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;

   // displayPeriod: the number of timesteps each input is displayed before switching to the next.
   // If this is <= 0 or inputPath does not end in .txt, assumes the input is a single file and will
   // not change.
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

   // start_frame_index: Array specifying the file indices to start at.
   // If displayPeriod <= 0, this determines which index from the file list will be used.
   virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);

   // skip_frame_index: Array specifying how much to increment the file index by each displayPeriod
   // for each batch
   virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);

   // writeFrameToTimestamp: if true, then every time the frame is updated, it writes the frame
   // number,
   // the time and the image filename to a file. The file is placed in a directory "timestamps" in
   // the outputPath
   // directory, and the filename is the layer name appended with ".txt".
   virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);

   // resetToStartOnLoop: If false, then when the end of file for the inputPath file is reached,
   // it rewinds to index 0. Otherwise, it rewinds to the index it began at (possibly
   // start_frame_index).
   virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);

   // batchMethod: Specifies how to split the file for batches.
   // byFile: Each batch skips nbatch, and starts staggered from the beginning of the file list
   // byList: Each batch skips 1, and starts at index = numFrames/numBatch
   // bySpecified: User specified start_frame_index and skip_frame_index, one for each batch
   // random: Randomizes the order of the given file. Does not duplicate indices until all are used
   virtual void ioParam_batchMethod(enum ParamsIOFlag ioFlag);

   // Random seed used when batchMethod == random.
   virtual void ioParam_randomSeed(enum ParamsIOFlag ioFlag);

   // useInputBCFlag: Specifies if the input should be scaled to fill margins
   virtual void ioParam_useInputBCflag(enum ParamsIOFlag ioFlag);

  protected:
   InputLayer() {}

   /**
    * This method scatters the mInputData buffer to the activity buffers of the several MPI
    * processes.
    */
   int scatterInput(int localBatchIndex, int mpiBatchIndex);
   int initialize(const char *name, HyPerCol *hc);

   // Returns PV_SUCCESS if offsetAnchor is a valid anchor string, PV_FAILURE otherwise.
   // (two characters long; first characters one of 't', 'c', or 'b'; second characters one of 'l',
   // 'c', or 'r')
   int checkValidAnchorString(const char *offsetAnchor);

   /**
    * normalizePixels transforms the input on a per-pixel basis specified by the batch element,
    * based on the normalizeLuminanceFlag, normalizeStdDev, and inverseFlag parameters.
    * Overload this to add additional post process steps in subclasses.
    * Pixels not occupied by the actual image (due to offsets, padding, etc.) are not changed.
    */
   virtual void normalizePixels(int batchElement);
   virtual void allocateV() override;
   virtual void initializeV() override;
   virtual void initializeActivity() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual double getDeltaUpdateTime() override;

   // Method that signals when to load the next file.
   // Can be overridden for different file list logic in subclasses.
   virtual bool readyForNextFile();

   /**
    * This pure virtual function gets called by initializeBatchIndexer in order
    * to give the BatchIndexer the number of input images.
    */
   virtual int countInputImages() = 0;

   /**
    * This pure virtual function gets called by the root process during
    * initializeActivity and during updateState. It loads the entire input
    * (scattering to nonroot processes is done by the scatterInput method)
    * into a buffer. inputIndex is the (zero-indexed) index into the list of inputs.
    */
   virtual Buffer<float> retrieveData(int inputIndex) = 0;

   /**
    * Each batch element loads its data with the input specified by the current
    * index for that batch element. The indices are not modified.
    */
   void retrieveInput(double timef, double dt);

   /**
    * Each batch element loads its input by calling retrieveInput(), and then
    * advances its index by the amount specified by skip_frame_index.
    */
   void retrieveInputAndAdvanceIndex(double timef, double dt);
   void initializeBatchIndexer();

  public:
   InputLayer(const char *name, HyPerCol *hc);
   virtual ~InputLayer();

   virtual int requireChannel(int channelNeeded, int *numChannelsResult) override;
   void makeInputRegionsPointer() { mNeedInputRegionsPointer = true; }
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double time, double dt) override;

   /**
    * This function is called by updateState if writeFrameToTimestamp is used.
    * It provides the opportunity for a subclass to insert a description of the
    * input, along with the timestamp and the name of the InputLayer.
    * The input argument is the index into the list of input frames.
    * The default is to return the empty string. However, ImageLayer overrides
    * this so that, when a list of filenames is used, it returns the name of
    * the filename for that index.
    */
   virtual std::string describeInput(int index) { return std::string(""); }
   virtual bool activityIsSpiking() override { return false; }
   int getDisplayPeriod() { return mDisplayPeriod; }
   int getStartIndex(int batchIndex) { return mStartFrameIndex.at(batchIndex); }
   int getSkipIndex(int batchIndex) { return mSkipFrameIndex.at(batchIndex); }
   const std::string &getInputPath() const { return mInputPath; }

   /**
    * A virtual method to return the filename containing the input of the current
    * batch index. Caveats: updateState advances the batch indices after loading
    * data, so this typically returns the filename that will belong to the *next*
    * input to be loaded, not the current input.
    * Since only the processes with MPIBlock rank zero do input/output,
    * the result of this method is only reliable for those processes.
    */
   virtual std::string const &getCurrentFilename(int localBatchElement, int mpiBatchIndex) const {
      return mInputPath;
   }

   float *getInputRegionsAllBatchElements() { return mInputRegionsAllBatchElements.data(); }

  private:
   /**
    * Resizes a buffer from the image size to the global layer size. If autoResizeFlag is true,
    * it calls BufferUtils::rescale. If autoResizeFlag is false, it calls Buffer methods grow,
    * translate, and crop. This method is called only by the root process.
    */
   void fitBufferToGlobalLayer(Buffer<float> &buffer, int blockBatchElement);

   void cropToMPIBlock(Buffer<float> &buffer);

  protected:
   // If mAutoResizeFlag is enabled, do we crop the edges or pad the edges with mPadValue?
   BufferUtils::RescaleMethod mRescaleMethod;

   // If mAutoResizeFlag is enabled, do we rescale with bicubic or nearest neighbor filtering?
   BufferUtils::InterpolationMethod mInterpolationMethod = BufferUtils::BICUBIC;

   // When cropping or resizing, which side of the canvas is the origin?
   Buffer<float>::Anchor mAnchor = Buffer<float>::CENTER;

   // Flag that enables rescaling input buffer to layer dimensions instead of just cropping
   bool mAutoResizeFlag = false;

   // Flag that inverts input buffer during post process step
   bool mInverseFlag = false;

   // Flag that enables scaling input buffer to extended region instead of restricted region
   bool mUseInputBCflag = false;

   // Flag enabling normalization in the post process step
   bool mNormalizeLuminanceFlag = false;

   // If true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
   // If false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
   bool mNormalizeStdDev = true;

   // Amount to translate input buffer before scattering but after rescaling
   int mOffsetX = 0;
   int mOffsetY = 0;

   // If nonzero, create a sample by shifting image randomly in [-maxRandomShiftX, maxRandomShiftX]
   // x [-maxRandomShiftY, maxRandomShiftY]
   int mMaxShiftX = 0;
   int mMaxShiftY = 0;
   // How often to change random shifts (measured in displayPeriods)
   int mJitterChangeInterval = 1;

   // Are horizontal or vertical mirror flips enabled during the augmentation stage?
   bool mXFlipEnabled = false;
   bool mYFlipEnabled = false;

   // If this is true, toggle mirror flips each time instead of randomly selecting
   bool mXFlipToggle = false;
   bool mYFlipToggle = false;

   // Random seed used when batchMethod == random
   int mRandomSeed = 123456789;

   // Object to handle assigning file indices to batch element
   std::unique_ptr<BatchIndexer> mBatchIndexer;
   BatchIndexer::BatchMethod mBatchMethod;

  private:
   // Data read from disk, one per batch element.
   std::vector<Buffer<float>> mInputData;

   // The parts of the mInputData buffers occupied by image data, as opposed to gaps created by
   // offsets or resizing with padding. When mInputData[b] is first filled using retrieveData,
   // mInputRegion[b] is created as a buffer of the same size filled with ones.
   // As operations that translate or resize are applied to mInputData[b], the same operation is
   // applied to mInputRegion[b]. When normalizePixels() normalizes or scatterInput() copies to the
   // clayer activity, only those pixels where mInputRegion[b] is nonzero are used.
   std::vector<Buffer<float>> mInputRegion;

   bool mNeedInputRegionsPointer = false;

   // A vector containing the contents of the mInputRegion buffers, allocated as a single array
   // of size getNumExtendedAllBatches.
   // Will not be allocated unless the makeInputRegionsPointer() method is called before the
   // AllocateData stage.
   std::vector<float> mInputRegionsAllBatchElements;

   // BorderExchange object for boundary exchange
   BorderExchange *mBorderExchanger = nullptr;

   // Value to fill empty region with when using padding
   float mPadValue = 0.0f;

   // Path to input file or list of input files
   std::string mInputPath;

   // Filepointer to output file used when mWriteFrameToTimestamp == true
   CheckpointableFileStream *mTimestampStream = nullptr;

   // Number of timesteps an input file is displayed before advancing the file list. If <= 0, the
   // file never changes.
   int mDisplayPeriod = 0;

   // When reaching the end of the file list, do we reset to 0 or to start_index?
   // This parameter is read only if using batchMethod=bySpecified
   bool mResetToStartOnLoop = false;

   // Flag to write filenames and batch indices to disk as they are loaded
   bool mWriteFrameToTimestamp = true;

   // An array of starting file list indices, one per batch
   std::vector<int> mStartFrameIndex;

   // An array indicating how far to advance each index, one per batch
   std::vector<int> mSkipFrameIndex;

   // Random number generator for jitter
   std::mt19937 mRNG;
   // An array of random shifts in x direction, one per batch
   std::vector<int> mRandomShiftX;
   // An array of random shifts in y direction, one per batch
   std::vector<int> mRandomShiftY;
   // Same for mirror flips
   std::vector<bool> mMirrorFlipX;
   std::vector<bool> mMirrorFlipY;
};

} // end namespace PV

#endif
