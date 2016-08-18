/*
 * InputLayer.cpp
 * Formerly InputLayer.cpp
 */

#include "InputLayer.hpp"

#include <algorithm>
#include <cfloat>

namespace PV {

   InputLayer::InputLayer() {
      initialize_base();
   }

   InputLayer::InputLayer(const char *name, HyPerCol *hc) {
      initialize_base();
      initialize(name, hc);
   }

   InputLayer::~InputLayer() {
      Communicator::freeDatatypes(mDatatypes);
   }

   int InputLayer::initialize_base() {
      mDatatypes = nullptr;
      mTimestampFile = nullptr;
      mUseInputBCflag = false;
      mAutoResizeFlag = false;
      mInterpolationMethod = Buffer::BICUBIC;
      mInverseFlag = false;
      mNormalizeLuminanceFlag = false;
      mNormalizeStdDev = true;
      mOffsetX = 0;
      mOffsetY = 0;
      mOffsetAnchor = Buffer::CENTER;
      mPadValue = 0;
      mEchoFramePathnameFlag = false;
      mDisplayPeriod = -1;
      mWriteFileToTimestamp = true;
      mResetToStartOnLoop = true;
      return PV_SUCCESS;
   }

   int InputLayer::initialize(const char * name, HyPerCol * hc) {
      int status = HyPerLayer::initialize(name, hc);
      this->lastUpdateTime = parent->getStartTime();
      PVParams * params = hc->parameters(); //What is the point of this?
      //Update on first timestep
      setNextUpdateTime(parent->simulationTime() + hc->getDeltaTime());

      if(mWriteFileToTimestamp){
         std::string timestampFilename = std::string(parent->getOutputPath()) + std::string("/timestamps/");
         parent->ensureDirExists(timestampFilename.c_str());
         timestampFilename += name + std::string(".txt");
         if(getParent()->getCommunicator()->commRank() == 0) {
             //If checkpoint read is set, append, otherwise, clobber
             if(getParent()->getCheckpointReadFlag()) {
                struct stat statbuf;
                if (PV_stat(timestampFilename.c_str(), &statbuf) != 0) {
                   pvWarn().printf("%s: timestamp file \"%s\" unable to be found.  Creating new file.\n",
                         getDescription_c(), timestampFilename.c_str());
                   mTimestampFile = PV::PV_fopen(timestampFilename.c_str(), "w", parent->getVerifyWrites());
                }
                else {
                   mTimestampFile = PV::PV_fopen(timestampFilename.c_str(), "r+", false/*verifyWrites*/);
                }
             }
             else{
                mTimestampFile = PV::PV_fopen(timestampFilename.c_str(), "w", parent->getVerifyWrites());
             }
             pvAssert(mTimestampFile);
         }
      }
      return status;
   }

   int InputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
      int status = HyPerLayer::ioParamsFillGroup(ioFlag);
      pvDebug() << "IOPARAMS ON RANK " << parent->getCommunicator()->commRank() << "\n";
      ioParam_inputPath(ioFlag);
      ioParam_offsetAnchor(ioFlag);
      ioParam_offsets(ioFlag);
      ioParam_autoResizeFlag(ioFlag);
      ioParam_aspectRatioAdjustment(ioFlag);
      ioParam_interpolationMethod(ioFlag);
      ioParam_inverseFlag(ioFlag);
      ioParam_normalizeLuminanceFlag(ioFlag);
      ioParam_normalizeStdDev(ioFlag);
      //ioParam_offsetConstraintMethod(ioFlag); //TODO: Reimplement this
      ioParam_useInputBCflag(ioFlag);
      ioParam_padValue(ioFlag);
      ioParam_displayPeriod(ioFlag);
      ioParam_echoFramePathnameFlag(ioFlag);
      ioParam_batchMethod(ioFlag);
      ioParam_start_frame_index(ioFlag);
      ioParam_skip_frame_index(ioFlag);
      ioParam_writeFrameToTimestamp(ioFlag);
      ioParam_resetToStartOnLoop(ioFlag);
      return status;
   }

   //What's the difference between this and checkpointRead?
   int InputLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
      if(parent->columnId() == 0) {
         int *frameNumbers;
         parent->readArrayFromFile(cpDir, getName(), "FrameNumbers", frameNumbers, parent->getNBatch());  
         std::vector<int> indices;
         indices.resize(parent->getNBatch());
         for(int n = 0; n < indices.size(); ++n) {
            indices.at(n) = frameNumbers[n];
         }
         mBatchIndexer->setIndices(indices);
         free(frameNumbers);
      }

      return PV_SUCCESS;
   }

   int InputLayer::checkpointRead(const char * cpDir, double * timef) {
      // should this be moved to readStateFromCheckpoint?
      if (mWriteFileToTimestamp) {
         long timestampFilePos = 0L;
         parent->readScalarFromFile(cpDir, getName(), "TimestampState", &timestampFilePos, timestampFilePos);
         if (mTimestampFile) {
            assert(parent->columnId() == 0);
            pvErrorIf(PV_fseek(mTimestampFile, timestampFilePos, SEEK_SET) != 0, "MovieLayer::checkpointRead error: unable to recover initial file position in timestamp file for layer %s: %s\n", name, strerror(errno));
         }
      }
      return HyPerLayer::checkpointRead(cpDir, timef);
   }
int InputLayer::checkpointWrite(const char * cpDir){
      if(parent->columnId() == 0) {

         parent->writeArrayToFile(cpDir, getName(), "FrameNumbers", static_cast<int*>(mBatchIndexer->getIndices().data()), parent->getNBatch());
      }
      //Only do a checkpoint TimestampState if there exists a timestamp file
      if (mTimestampFile) {
         long timestampFilePos = getPV_StreamFilepos(mTimestampFile);
         parent->writeScalarToFile(cpDir, getName(), "TimestampState", timestampFilePos);
      }
      return HyPerLayer::checkpointWrite(cpDir);
   }

   double InputLayer::getDeltaUpdateTime() { 
      return mDisplayPeriod > 0 ? mDisplayPeriod : DBL_MAX;
   }

   void InputLayer::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
      // TODO: Make sure other string params are handled correctly
      char *tempString = nullptr;
      if(ioFlag == PARAMS_IO_WRITE) {
         tempString = (char*)calloc(mInputPath.size() + 1, sizeof(char));
         tempString = strcpy(tempString, mInputPath.c_str());
      }
      parent->ioParamStringRequired(ioFlag, name, "inputPath", &tempString);
      if(ioFlag == PARAMS_IO_READ) { 
         mInputPath = std::string(tempString);
         // Check if the input path ends in ".txt" and enable the file list if so
         std::string txt = ".txt";
         if(mInputPath.size() > txt.size() && mInputPath.compare(mInputPath.size() - txt.size(), txt.size(), txt) == 0) {
            mUsingFileList = true; //TODO: Add a flag to override this value even when the input path ends in ".txt"
         }
         else {
            mUsingFileList = false;
         }
      }
      free(tempString);
   }

   void InputLayer::ioParam_useInputBCflag(enum ParamsIOFlag ioFlag) { //TODO: Change to useInputBCFlag, add deprecated warning
      parent->ioParamValue(ioFlag, name, "useImageBCflag", &mUseInputBCflag, mUseInputBCflag);
   }

   int InputLayer::ioParam_offsets(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "offsetX", &mOffsetX, mOffsetX);
      parent->ioParamValue(ioFlag, name, "offsetY", &mOffsetY, mOffsetY);

      return PV_SUCCESS;
   }

   void InputLayer::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag){
      if (ioFlag==PARAMS_IO_READ) {
         char *offsetAnchor = nullptr;
         parent->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
         if(strcmp(offsetAnchor, "tl") == 0) {
            mOffsetAnchor = Buffer::NORTHWEST;
         }
         else if(strcmp(offsetAnchor, "tc") == 0) {
            mOffsetAnchor = Buffer::NORTH;
         }
         else if(strcmp(offsetAnchor, "tr") == 0) {
            mOffsetAnchor = Buffer::NORTHEAST;
         }
         else if(strcmp(offsetAnchor, "cl") == 0) {
            mOffsetAnchor = Buffer::WEST;
         }
         else if(strcmp(offsetAnchor, "cc") == 0) {
            mOffsetAnchor = Buffer::CENTER;
         }
         else if(strcmp(offsetAnchor, "cr") == 0) {
            mOffsetAnchor = Buffer::EAST;
         }
         else if(strcmp(offsetAnchor, "bl") == 0) { 
            mOffsetAnchor = Buffer::SOUTHWEST;
         }
         else if(strcmp(offsetAnchor, "bc") == 0) {
            mOffsetAnchor = Buffer::SOUTH;
         }
         else if(strcmp(offsetAnchor, "br") == 0) {
            mOffsetAnchor = Buffer::SOUTHEAST;
         }
         else {
            if (parent->columnId()==0) {
               pvErrorNoExit().printf("%s: offsetAnchor must be a two-letter string.  The first character must be \"t\", \"c\", or \"b\" (for top, center or bottom); and the second character must be \"l\", \"c\", or \"r\" (for left, center or right).\n", getDescription_c());
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
         free(offsetAnchor);
      }
      else { //Writing
         //The opposite of above. Find a better way to do this that isn't so gross
         char *offsetAnchor = (char*)calloc(3, sizeof(char));
         offsetAnchor[2] = '\0';
         switch(mOffsetAnchor) {
            case Buffer::NORTH:
            case Buffer::NORTHWEST:
            case Buffer::NORTHEAST:
               offsetAnchor[0] = 't';
               break;
            case Buffer::WEST:
            case Buffer::CENTER:
            case Buffer::EAST:
               offsetAnchor[0] = 'c';
               break;
            case Buffer::SOUTHWEST:
            case Buffer::SOUTH:
            case Buffer::SOUTHEAST:
               offsetAnchor[0] = 'b';
               break;
         }
         switch(mOffsetAnchor) {
            case Buffer::NORTH:
            case Buffer::CENTER:
            case Buffer::SOUTH:
               offsetAnchor[1] = 'c';
               break;
            case Buffer::EAST:
            case Buffer::NORTHEAST:
            case Buffer::SOUTHEAST:
               offsetAnchor[1] = 'l';
               break;
            case Buffer::WEST:
            case Buffer::NORTHWEST:
            case Buffer::SOUTHWEST:
               offsetAnchor[1] = 'r';
               break;
         }
         parent->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
         free(offsetAnchor);
      }
   }

   void InputLayer::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "autoResizeFlag", &mAutoResizeFlag, mAutoResizeFlag);
   }

   void InputLayer::ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      if (mAutoResizeFlag) {
         char *aspectRatioAdjustment;
         parent->ioParamString(ioFlag, name, "aspectRatioAdjustment", &aspectRatioAdjustment, "crop"/*default*/);
         if (ioFlag == PARAMS_IO_READ) {
            assert(aspectRatioAdjustment);
            for (char * c = aspectRatioAdjustment; *c; c++) { *c = tolower(*c); }
         }
         if(strcmp(aspectRatioAdjustment, "crop") == 0) {
            mRescaleMethod = Buffer::CROP;
         }
         else if(strcmp(aspectRatioAdjustment, "pad") == 0) {
            mRescaleMethod = Buffer::PAD;
         }
         else {
            if (parent->columnId()==0) {
               pvErrorNoExit().printf("%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                     getDescription_c());
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
         free(aspectRatioAdjustment);
      }
   }

   void InputLayer::ioParam_interpolationMethod(enum ParamsIOFlag ioFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      if (mAutoResizeFlag) {
         char * interpolationMethodString = NULL;
         if (ioFlag == PARAMS_IO_READ) {
            parent->ioParamString(ioFlag, name, "interpolationMethod", &interpolationMethodString, "bicubic", true/*warn if absent*/);
            assert(interpolationMethodString);
            for (char * c = interpolationMethodString; *c; c++) { *c = tolower(*c); }
            if (!strncmp(interpolationMethodString, "bicubic", strlen("bicubic"))) {
               mInterpolationMethod = Buffer::BICUBIC;
            }
            else if (!strncmp(interpolationMethodString, "nearestneighbor", strlen("nearestneighbor"))) {
               mInterpolationMethod = Buffer::NEAREST;
            }
            else {
               if (parent->columnId()==0) {
                  pvErrorNoExit().printf("%s: interpolationMethod must be either \"bicubic\" or \"nearestNeighbor\".\n",
                        getDescription_c());
               }
               MPI_Barrier(parent->getCommunicator()->communicator());
               exit(EXIT_FAILURE);
            }
         }
         else {
            assert(ioFlag == PARAMS_IO_WRITE);
            switch (mInterpolationMethod) {
            case Buffer::BICUBIC:
               interpolationMethodString = strdup("bicubic");
               break;
            case Buffer::NEAREST:
               interpolationMethodString = strdup("nearestNeighbor");
               break;
            default:
               assert(0); // interpolationMethod should be one of the above two categories.
            }
            parent->ioParamString(ioFlag, name, "interpolationMethod", &interpolationMethodString, "bicubic", true/*warn if absent*/);
         }
         free(interpolationMethodString);
      }
   }

   void InputLayer::ioParam_inverseFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "inverseFlag", &mInverseFlag, mInverseFlag);
   }

   void InputLayer::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "normalizeLuminanceFlag", &mNormalizeLuminanceFlag, mNormalizeLuminanceFlag);
   }

   void InputLayer::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeLuminanceFlag"));
      if (mNormalizeLuminanceFlag) {
        parent->ioParamValue(ioFlag, name, "normalizeStdDev", &mNormalizeStdDev, mNormalizeStdDev);
      }
   }
   void InputLayer::ioParam_padValue(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "padValue", &mPadValue, mPadValue);
   }
   
//   void InputLayer::ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag) {
//   //   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
//   //   if (mJitterFlag) {
//         parent->ioParamValue(ioFlag, name, "offsetConstraintMethod", &mOffsetConstraintMethod, 0/*default*/);
//         if (ioFlag == PARAMS_IO_READ && (mOffsetConstraintMethod <0 || mOffsetConstraintMethod >3) ) {
//            pvError().printf("%s: offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getDescription_c());
//         }
//   //   }
//   }

   void InputLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
      assert(this->initVObject == NULL);
      return;
   }

   void InputLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
      if (ioFlag == PARAMS_IO_READ) {
         triggerLayerName = NULL;
         triggerFlag = false;
         parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL/*correct value*/);
      }
   }

   void InputLayer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "displayPeriod", &mDisplayPeriod, mDisplayPeriod);
   }

   void InputLayer::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &mEchoFramePathnameFlag, false/*default value*/);
   }

   void InputLayer::ioParam_batchMethod(enum ParamsIOFlag ioFlag) {
      if(!mUsingFileList) {
         return;
      }
      char *batchMethod = (char*)calloc(sizeof(char), 256);
      parent->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "byFile");
      if(strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byFile") == 0) {
         mBatchMethod = BatchIndexer::BYFILE;
      }
      else if(strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "byList") == 0) {
         mBatchMethod = BatchIndexer::BYLIST;
      } 
      else if(strcmp(batchMethod, "bySpecified") == 0) {
         mBatchMethod = BatchIndexer::BYSPECIFIED;
      }
      else{
         pvError() << "WARNING: Input layer " << name << " batchMethod not recognized. Options are \"byFile\", \"byList\", and \"bySpecified\"\n.";
      }
      free(batchMethod);
   }

   void InputLayer::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
      if(ioFlag == PARAMS_IO_READ) { // TODO: Fix writing out arrays
         int *paramsStartFrameIndex;
         int length = -1;
         this->getParent()->ioParamArray(ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &length);
         mStartFrameIndex.clear();
         mStartFrameIndex.resize(parent->getNBatch());
         if(length > 0) {
            for(int i = 0; i < length; ++i) {
               mStartFrameIndex.at(i) = paramsStartFrameIndex[i];
            }
         }
         free(paramsStartFrameIndex);
      }
   }

   void InputLayer::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
      if(ioFlag == PARAMS_IO_READ) { // TODO: Same as above
         int *paramsSkipFrameIndex;
         int length = -1;
         this->getParent()->ioParamArray(ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &length);
         mSkipFrameIndex.clear();
         mSkipFrameIndex.resize(parent->getNBatch());
         if(length > 0) {
            for(int i = 0; i < length; ++i) {
               mSkipFrameIndex.at(i) = paramsSkipFrameIndex[i];
            }
         }
         free(paramsSkipFrameIndex);
      }
   }

   void InputLayer::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "writeFrameToTimestamp", &mWriteFileToTimestamp, mWriteFileToTimestamp);
   }

   void InputLayer::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "resetToStartOnLoop", &mResetToStartOnLoop, mResetToStartOnLoop);
   }

   int InputLayer::allocateDataStructures() {
      int status = HyPerLayer::allocateDataStructures();
      if(status != PV_SUCCESS) {
         return status;
      }
      pvDebug() << "ALLOCATE STARTING RANK " << parent->getCommunicator()->commRank() << "\n";
      int numBatch = parent->getNBatch();
 
      //Calculate file positions for beginning of each frame
      if(mUsingFileList) {
         populateFileList();
         pvInfo() << "File " << mInputPath << " contains " << mFileList.size() << " frames\n";

      }

      if(parent->columnId() == 0) {
         initializeBatchIndexer(mFileList.size());
      }

      // TODO: Should this only happen on root?
      mInputData.resize(numBatch);
      for(int b = 0; b < numBatch; ++b) {
         mInputData.at(b).resize(getLayerLoc()->ny, getLayerLoc()->nx, getLayerLoc()->nf);
      }
      nextInput(parent->simulationTime(), parent->getDeltaTimeBase());
      pvDebug() << "ALLOCATE ENDING RANK " << parent->getCommunicator()->commRank() << ", " << getName() << "\n";
      
      // create mpi_datatypes for border transfer
      mDatatypes = Communicator::newDatatypes(getLayerLoc());
      exchange();
      pvDebug() << "EXCHANGE COMPLETED RANK " << parent->getCommunicator()->commRank() << ", " << getName() << "\n";
      return status;
   }

   void InputLayer::initializeBatchIndexer(int fileCount) {
      pvDebug() << "INIT BATCH WITH COUNT " << fileCount << "\n";
      mBatchIndexer = std::unique_ptr<BatchIndexer>(new BatchIndexer(
               parent->getNBatchGlobal(),
               parent->commBatch() * parent->getNBatch(),
               parent->numCommBatches(),
               fileCount,
               mBatchMethod));
      for(int b = 0; b < parent->getNBatch(); ++b) {
         mBatchIndexer->specifyBatching(b, mStartFrameIndex.at(b), mSkipFrameIndex.at(b));
         mBatchIndexer->initializeBatch(b);
      }
   }

   double InputLayer::calcTimeScale(int batchIdx) {
      if(needUpdate(parent->simulationTime(), parent->getDeltaTime())) {
         return parent->getTimeScaleMin(); 
      }
      else {
         return HyPerLayer::calcTimeScale(batchIdx);
      }
   }

   // Virtual method used to spend multiple display periods on one file.
   // Can be used to implement lists of collections or modifications to
   // the loaded file, such as streaming audio or video.
   bool InputLayer::readyForNextFile() { 
      // A display period <= 0 means we never change files
      return mDisplayPeriod > 0;
   }

   int InputLayer::updateState(double time, double dt)  {
      // TODO: We might still want to update the state on a single file, like a MoviePvp.
      // Figure out the best way to accomplish this
      //if(!mUsingFileList) {
      //   return PV_SUCCESS;
      //}

      Communicator * icComm = getParent()->getCommunicator();
      //Only do this if it's not the first update timestep
      //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
      //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).
//      if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)) {
         pvDebug() << "UPDATE STATE CALLED BY RANK " << parent->getCommunicator()->commRank() << ", " << getName() << "\n";
         if(readyForNextFile()) {
            nextInput(time, dt);
            //Write to timestamp file 
            if(icComm->commRank() == 0) {
               if(mTimestampFile) {
                  std::ostringstream outStrStream;
                  outStrStream.precision(15);
                  int kb0 = getLayerLoc()->kb0;
                  for(int b = 0; b < parent->getNBatch(); ++b) {
//                     outStrStream << time << "," << b+kb0 << "," << mFileIndices.at(b) << "," << mFileList.at(b) << "\n";
                  }
                  size_t len = outStrStream.str().length();
                  int status = PV_fwrite(outStrStream.str().c_str(), sizeof(char), len, mTimestampFile) == len ? PV_SUCCESS : PV_FAILURE;
                  pvErrorIf(status != PV_SUCCESS, "%s: Movie::updateState failed to write to timestamp file.\n", getDescription_c());
                  fflush(mTimestampFile->fp);
               }
            }
         }
      return PV_SUCCESS;
   }

   void InputLayer::nextInput(double timef, double dt) {
      pvDebug() << "NEXTINPUT CALLED BY RANK " << parent->getCommunicator()->commRank() << ", " << getName() << "\n";
      for(int b = 0; b < parent->getNBatch(); b++) {
         if (parent->columnId() == 0) {
            std::string fileName = mInputPath;
            if(mUsingFileList) {
               // TODO: This needs to use global batch index, not local. Fix it
               fileName = mFileList.at(mBatchIndexer->nextIndex(b));
               //fileName = getNextFilename(mSkipFrameIndex.at(b), b);
            }
            mInputData.at(b) = retrieveData(fileName, b);
         } 
         scatterInput(b);
      }
      //postProcess(timef, dt);
   }

   int InputLayer::scatterInput(int batchIndex) {
      Communicator *icComm = parent->getCommunicator();
      const int rank = parent->columnId();
      const int rootProc = 0;
      MPI_Comm mpiComm = parent->getCommunicator()->communicator();
      pvadata_t *activityBuffer = getActivity() + batchIndex * getNumExtended();
      const PVLayerLoc *loc = getLayerLoc();
      const PVHalo *halo = &loc->halo;
      const int numFeatures = loc->nf;
      int activityWidth = loc->nx + (mUseInputBCflag ? halo->lt + halo->rt : 0);
      int activityHeight = loc->ny + (mUseInputBCflag ? halo->up + halo->dn : 0);
      int numElements = activityHeight * activityWidth * numFeatures;

      // Defining this outside of the loop lets it contain the correct
      // data for the root process at the end
      Buffer croppedBuffer;
      pvDebug() << "BEGINNING SCATTER ON RANK " << rank << " AWIDTH=" << activityWidth << " AHEIGHT=" << activityHeight << "\n";      
      if (rank == rootProc) {

         // Loop through each rank, ending on the root process.
         // Uses Buffer::crop and MPI_Send to give each process the correct slice
         // of input data. Once a process has the data, it slices it up row by row
         // as is needed by mUseInputBCflag
         for (int rank = icComm->commSize()-1; rank >= 0; --rank) {
            
            // Copy the input data to a temporary buffer. This gets cropped to the layer size below.
            croppedBuffer = mInputData.at(batchIndex);
            int cropLeft = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns()) * loc->nx;
            int cropTop = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns()) * loc->ny;

            // If we're sending the extended region as well, shift our origin by the appropriate amount
            // TODO: This is going to give negative indices for some slices. What's the correct approach here?
            // E: Pretty sure this resolves itself, making this unneccessary. Uncomment and investigate further if not.
            // if(mUseInputBCflag) {
            //    cropLeft -= halo->lt;
            //    cropTop -= halo->up;
            // }

            // Crop the input data to the size of one process.
            croppedBuffer.crop(activityWidth, activityHeight, Buffer::NORTHWEST, cropLeft, cropTop);

            // If this isn't the root process, ship it off to the appropriate process.
            if(rank != rootProc) {
               // This is required because croppedBuffer.asVector() returns a const vector<>
               std::vector<float> bufferData = croppedBuffer.asVector();
               pvDebug() << "SCATTERING FROM RANK 0 TO RANK " << rank << ", " << numElements << " ELEMENTS\n";
               MPI_Send(bufferData.data(), numElements, MPI_FLOAT, rank, 31, mpiComm);
            }
         }
      }
      else {
         pvDebug() << "RECEIVING ON RANK " << parent->getCommunicator()->commRank() << "\n";
         // Create a temporary array to receive from MPI, move the values into
         // a vector, and then create a Buffer out of that vector. A little
         // redundant, but it works. This could be done with a for loop and
         // some indexing math, but it's safer to let Buffer handle this
         // internally.
         float tempBuffer[numElements];
         MPI_Recv(&tempBuffer, numElements, MPI_FLOAT, rootProc, 31, mpiComm, MPI_STATUS_IGNORE);
         std::vector<float> bufferData(numElements);
         for(int i = 0; i < numElements; ++i) {
            bufferData.at(i) = tempBuffer[i];
         }
         croppedBuffer.set(bufferData, activityWidth, activityHeight, numFeatures);
      }
      
      pvDebug() << "COPYING DATA TO ACTIVITY ON RANK " << parent->getCommunicator()->commRank() << "\n";

      // At this point, croppedBuffer has the correct data for this
      // process, regardless of if we are root or not. Clear the current
      // activity buffer, then copy the input data over row by row.
      for (int n = 0; n < getNumExtended(); ++n) {
         activityBuffer[n] = mPadValue;
      }
      for(int y = 0; y < activityHeight; ++y) {
         for(int x = 0; x < activityWidth; ++x) {
            for(int f = 0; f < numFeatures; ++f) {
               int activityIndex = kIndex(
                           halo->lt + x,
                           halo->up + y, 
                           f,
                           loc->nx+halo->lt+halo->rt,
                           loc->ny+halo->up+halo->dn,
                           numFeatures
                           );
               activityBuffer[activityIndex] = croppedBuffer.at(x, y, f);
            }
         }
      }
//         memcpy(&activityBuffer[destStart], &croppedBuffer.asVector().data()[sourceStart], 2 * sizeof(float)); //TODO: Verify this memcpy and see if there's a C++11 alternative
      

      // TODO:
      // Do I need to store any info about the image now? I don't belive so,
      // since it's the same size as the layer now, but verify

      pvDebug() << "SCATTER COMPLETED ON RANK " << rank << "\n";
      return PV_SUCCESS;
   }

   void InputLayer::fitBufferToLayer(Buffer &buffer) {
      pvAssert(parent->columnId() == 0); // Should only be called by root process.

      const PVLayerLoc *loc = getLayerLoc();
      const PVHalo *halo = &loc->halo;
      const int targetWidth = loc->nxGlobal + (mUseInputBCflag ? (halo->lt + halo->rt) : 0);
      const int targetHeight = loc->nyGlobal + (mUseInputBCflag ? (halo->dn + halo->up) : 0);
 
      if(mAutoResizeFlag) {
         buffer.rescale(targetWidth, targetHeight, mRescaleMethod, mInterpolationMethod); 
      }
      buffer.crop(targetWidth, targetHeight, mOffsetAnchor, mOffsetX, mOffsetY); 
   }     

   //Apply normalizeLuminanceFlag, normalizeStdDev, and inverseFlag, which can be done pixel-by-pixel
   //after scattering.
   int InputLayer::postProcess(double timef, double dt){
      int numExtended = getNumExtended();

      // if normalizeLuminanceFlag == true:
      //     if normalizeStdDev is true, then scale so that average luminance to be 0 and std. dev. of luminance to be 1.
      //     if normalizeStdDev is false, then scale so that minimum is 0 and maximum is 1
      // if normalizeLuminanceFlag == true and the image in buffer is completely flat, force all values to zero
      for(int b = 0; b < parent->getNBatch(); b++){
         float* buf = getActivity() + b * numExtended;
         if(mNormalizeLuminanceFlag){
            if (mNormalizeStdDev){
               double image_sum = 0.0f;
               double image_sum2 = 0.0f;
               for (int k=0; k<numExtended; k++) {
                  image_sum += buf[k];
                  image_sum2 += buf[k]*buf[k];
               }
               double image_ave = image_sum / numExtended;
               double image_ave2 = image_sum2 / numExtended;
#ifdef PV_USE_MPI
               MPI_Allreduce(MPI_IN_PLACE, &image_ave, 1, MPI_DOUBLE, MPI_SUM, parent->getCommunicator()->communicator());
               image_ave /= parent->getCommunicator()->commSize();
               MPI_Allreduce(MPI_IN_PLACE, &image_ave2, 1, MPI_DOUBLE, MPI_SUM, parent->getCommunicator()->communicator());
               image_ave2 /= parent->getCommunicator()->commSize();
#endif // PV_USE_MPI
               // set mean to zero
               for (int k=0; k<numExtended; k++) {
                  buf[k] -= image_ave;
               }
               // set std dev to 1
               double image_std = sqrt(image_ave2 - image_ave*image_ave); // std = 1/N * sum((x[i]-sum(x[i])^2) ~ 1/N * sum(x[i]^2) - (sum(x[i])/N)^2 | Note: this permits running std w/o needing to first calc avg (although we already have avg)
               if(image_std == 0){
                  for (int k=0; k<numExtended; k++) {
                     buf[k] = 0.0;
                  }
               }
               else{
                  for (int k=0; k<numExtended; k++) {
                     buf[k] /= image_std;
                  }
               }
            }
            else{
               float image_max = -FLT_MAX;
               float image_min = FLT_MAX;
               for (int k=0; k<numExtended; k++) {
                  image_max = buf[k] > image_max ? buf[k] : image_max;
                  image_min = buf[k] < image_min ? buf[k] : image_min;
               }
               MPI_Allreduce(MPI_IN_PLACE, &image_max, 1, MPI_FLOAT, MPI_MAX, parent->getCommunicator()->communicator());
               MPI_Allreduce(MPI_IN_PLACE, &image_min, 1, MPI_FLOAT, MPI_MIN, parent->getCommunicator()->communicator());
               if (image_max > image_min){
                  float image_stretch = 1.0f / (image_max - image_min);
                  for (int k=0; k<numExtended; k++) {
                     buf[k] -= image_min;
                     buf[k] *= image_stretch;
                  }
               }
               else{ // image_max == image_min, set to gray
                  for (int k=0; k<numExtended; k++) {
                     buf[k] = 0.0f;
                  }
               }
            }
         } // normalizeLuminanceFlag
         if(mInverseFlag) {
            for (int k=0; k<numExtended; k++) {
               buf[k] = 1 - buf[k]; // If normalizeLuminanceFlag is true, should the effect of inverseFlag be buf[k] = -buf[k]?
            }
         }
      }
      return PV_SUCCESS;
   }

   void InputLayer::exchange() {
      std::vector<MPI_Request> req{};
      for (int b=0; b<getLayerLoc()->nbatch; ++b) {
         parent->getCommunicator()->exchange(getActivity()+b*getNumExtended(), mDatatypes, getLayerLoc(), req);
         parent->getCommunicator()->wait(req);
         pvAssert(req.empty());
      }
   }

   int InputLayer::requireChannel(int channelNeeded, int * numChannelsResult) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s cannot be a post-synaptic layer.\n",
               getDescription_c());
      }
      *numChannelsResult = 0;
      return PV_FAILURE;
   }

   int InputLayer::allocateV() {
      clayer->V = nullptr;
      return PV_SUCCESS;
   }

   int InputLayer::initializeV() {
      pvAssert(getV() == nullptr);
      return PV_SUCCESS;
   }

   int InputLayer::initializeActivity() {
      return PV_SUCCESS;
   }

   void InputLayer::populateFileList() {
      if(mUsingFileList && parent->columnId() == 0) {
         std::string line;
         mFileList.clear();
         pvInfo() << "Reading list: " << mInputPath << "\n";
         std::ifstream infile(mInputPath, std::ios_base::in);
         while(getline(infile, line, '\n')) {
            std::string noWhiteSpace = line;
            noWhiteSpace.erase(std::remove_if(noWhiteSpace.begin(), noWhiteSpace.end(), ::isspace), noWhiteSpace.end());
            if(!noWhiteSpace.empty()) {
               pvInfo() << noWhiteSpace << "\n";
               mFileList.push_back(noWhiteSpace);
            }
         }
      }
   }
} 





