// InputLayer
// Base class for layers that take their input from file IO

#pragma once

#include "HyPerLayer.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/Buffer.hpp"
#include "utils/BatchIndexer.hpp"

#include <memory>

namespace PV {

   class InputLayer: public HyPerLayer {
      protected:
         
         // inputPath: Either an individual file to load, or a .txt list of files to load.
         virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag);

         // offsetX: offset in X direction
         // offsetY: offset in Y direction
         // Defines an offset in image space where the column is viewing the image
         virtual int ioParam_offsets(enum ParamsIOFlag ioFlag); 
         
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

         // inverseFlag: If set to true, inverts the input 
         virtual void ioParam_inverseFlag(enum ParamsIOFlag ioFlag);

         // normalizeLuminanceFlag: If set to true, will normalize the image.
         // The normalization method is determined by the normalizeStdDev parameter.
         virtual void ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag);
         
         // normalizeStdDev: This flag is used if normalizeLuminanceFlag is true.
         // If normalizeStdDev is set to true, the image will normalize with a mean of 0 and std of 1
         // If normalizeStdDev is set to false, the image will normalize with a min of 0 and a max of 1
         virtual void ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag);

         // padValue: If the image is being padded (image smaller than layer), the value to use for padding
         virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

         // initVType: InputLayers do not have a V, do not set
         virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

         // triggerLayerName: InputLayer and derived classes do not use triggering, and always set triggerLayerName to NULL.
         virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

         // displayPeriod: the number of timesteps each input is displayed before switching to the next.
         // If this is <= 0 or inputPath does not end in .txt, assumes the input is a single file and will not change.
         virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

         // echoFramePathnameFlag: if true, print the filename when a new file is loaded.
         virtual void ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag);

         // start_frame_index: Array specifying the file indices to start at.
         // If displayPeriod <= 0, this determines which index from the file list will be used.
         virtual void ioParam_start_frame_index(enum ParamsIOFlag ioFlag);

         // skip_frame_index: Array specifying how much to increment the file index by each displayPeriod for each batch
         virtual void ioParam_skip_frame_index(enum ParamsIOFlag ioFlag);

         // writeFrameToTimestamp: if true, then every time the frame is updated, it writes the frame number,
         // the time and the image filename to a file. The file is placed in a directory "timestamps" in the outputPath
         // directory, and the filename is the layer name appended with ".txt".
         virtual void ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag);

         // resetToStartOnLoop: If false, then when the end of file for the inputPath file is reached,
         // it rewinds to index 0. Otherwise, it rewinds to the index it began at (possibly start_frame_index).
         virtual void ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag);
         
         // batchMethod: Specifies how to split the file for batches.
         // byFile: Each batch skips nbatch, and starts staggered from the beginning of the file list
         // byList: Each batch skips 1, and starts at index = numFrames/numBatch
         // bySpecified: User specified start_frame_index and skip_frame_index, one for each batch
         virtual void ioParam_batchMethod(enum ParamsIOFlag ioFlag);
         
         //useImageBCFlag: Specifies if the input should be scaled to fill margins 
         virtual void ioParam_useInputBCflag(enum ParamsIOFlag ioFlag);

      protected:
         InputLayer();

         // This method scatters the mInputData buffer to the activity buffers of the several MPI processes.
         int scatterInput(int batchIndex);
         int initialize(const char * name, HyPerCol * hc);
       
         // Returns PV_SUCCESS if offsetAnchor is a valid anchor string, PV_FAILURE otherwise.
         // (two characters long; first characters one of 't', 'c', or 'b'; second characters one of 'l', 'c', or 'r')
         int checkValidAnchorString();
 
         // This method post processes the activity buffer after a file is loaded and scattered.
         // Overload this to add additional post process steps in subclasses.
         virtual int postProcess(double timef, double dt);
         virtual int allocateV();
         virtual int initializeV();
         virtual int initializeActivity();
         virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
         virtual int readStateFromCheckpoint(const char* cpDir, double* timeptr);
         virtual double getDeltaUpdateTime();

         // Method that signals when to load the next file. 
         // Can be overridden for different file list logic in subclasses.
         virtual bool readyForNextFile();

         // This pure virtual function gets called from nextInput by the root process only.
         // Load the input file from disk in this method.
         virtual Buffer retrieveData(std::string filename, int batchIndex) = 0;
         void nextInput(double timef, double dt);
         void initializeBatchIndexer(int fileCount);

      public:
         InputLayer(const char * name, HyPerCol * hc);
         virtual ~InputLayer();

         virtual int requireChannel(int channelNeeded, int * numChannelsResult);
         virtual int allocateDataStructures();
         virtual int updateState(double time, double dt);
         virtual int checkpointRead(const char * cpDir, double * timef);
         virtual int checkpointWrite(const char *cpDir);
         virtual double calcTimeScale(int batchIndex);
         virtual bool activityIsSpiking() {return false;}
         void exchange();
         int getDisplayPeriod() { return mDisplayPeriod; }
         int getStartIndex(int batchIndex) { return mStartFrameIndex.at(batchIndex); }
         int getSkipIndex(int batchIndex) { return mSkipFrameIndex.at(batchIndex); }
         const std::string getInputPath() { return mInputPath; }

      private:
         int initialize_base();
         void populateFileList();
         void fitBufferToLayer(Buffer &buffer);

      protected:
         // Raw data read from disk, one per batch
         std::vector<Buffer> mInputData;
         // Enums that specify how fitBufferToLayer does its job
         Buffer::RescaleMethod mRescaleMethod;
         Buffer::InterpolationMethod mInterpolationMethod;
         Buffer::OffsetAnchor mOffsetAnchor;
         bool mAutoResizeFlag;
         bool mInverseFlag;
         bool mUseInputBCflag;
         bool mNormalizeLuminanceFlag;
         // if true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
         // if false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
         bool mNormalizeStdDev;
         int mOffsetX = 0;
         int mOffsetY = 0;
         // Object to handle batch indexing. 
         std::unique_ptr<BatchIndexer> mBatchIndexer;
         BatchIndexer::BatchMethod mBatchMethod;

      private:
         // MPI datatypes for boundary exchange
         MPI_Datatype* mDatatypes = nullptr;
         float mPadValue = 0.0f;
         std::string mInputPath;
         PV_Stream* mTimestampFile = nullptr;
         // Number of timesteps an input file is displayed before advancing the file list. If <= 0, the file never changes.
         int mDisplayPeriod = 0;
         // Automatically set if the inputPath ends in .txt. Determines whether this layer represents a collection of files.
         bool mUsingFileList = false;
         // if true, echo the frame pathname to output stream
         bool mEchoFramePathnameFlag = false;          
         // When reaching the end of the file list, do we reset to 0 or to start_index?
         bool mResetToStartOnLoop = false;
         bool mWriteFileToTimestamp = false;
         // An array of starting file list indices, one per batch
         std::vector<int> mStartFrameIndex;
         // An array indicating how far to advance each index, one per batch
         std::vector<int> mSkipFrameIndex;
         // List of filenames to iterate over
         std::vector<std::string> mFileList;
   };
} 
