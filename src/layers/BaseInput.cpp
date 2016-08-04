/*
 * BaseInput.cpp
 */

#include "BaseInput.hpp"

#include <algorithm>

namespace PV {

   BaseInput::BaseInput() {
      initialize_base();
   }

   BaseInput::~BaseInput() {
      Communicator::freeDatatypes(mDatatypes);
      free(mAspectRatioAdjustment);
      if(mOffsetAnchor){
         free(mOffsetAnchor);
      }
   }

   int BaseInput::initialize_base() {
      mDatatypes = nullptr;
      mUseInputBCflag = false;
      mAutoResizeFlag = false;
      mAspectRatioAdjustment = nullptr;
      mInterpolationMethod = Buffer::BICUBIC;
      mInverseFlag = false;
      mNormalizeLuminanceFlag = false;
      mNormalizeStdDev = true;
      mOffsets[0] = 0;
      mOffsets[1] = 0;
      mOffsetAnchor = nullptr;
      mPadValue = 0;
      mEchoFramePathnameFlag = false;
      mDisplayPeriod = 1;
      mWriteFileToTimestamp = true;
      mResetToStartOnLoop = true;
      return PV_SUCCESS;
   }

   int BaseInput::initialize(const char * name, HyPerCol * hc) {
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

   int BaseInput::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
      int status = HyPerLayer::ioParamsFillGroup(ioFlag);
      ioParam_inputPath(ioFlag);
      ioParam_offsetAnchor(ioFlag);
      ioParam_offsets(ioFlag);
      ioParam_autoResizeFlag(ioFlag);
      ioParam_aspectRatioAdjustment(ioFlag);
      ioParam_interpolationMethod(ioFlag);
      ioParam_inverseFlag(ioFlag);
      ioParam_normalizeLuminanceFlag(ioFlag);
      ioParam_normalizeStdDev(ioFlag);
      ioParam_offsetConstraintMethod(ioFlag);
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
   int BaseInput::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
      int *frameNumbers;
      parent->readArrayFromFile(cpDir, getName(), "FrameNumbers", frameNumbers, parent->getNBatch());
      mFileIndices.clear();
      mFileIndices.resize(parent->getNBatch());
      for(int n = 0; n < mFileIndices.size(); ++n) {
         mFileIndices.at(n) = frameNumbers[n];
      }
      free(frameNumbers);
      return PV_SUCCESS;
   }

   int BaseInput::checkpointRead(const char * cpDir, double * timef) {
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

   int BaseInput::checkpointWrite(const char * cpDir){
      parent->writeArrayToFile(cpDir, getName(), "FrameNumbers", &mFileIndices[0], parent->getNBatch());

      //Only do a checkpoint TimestampState if there exists a timestamp file
      if (mTimestampFile) {
         long timestampFilePos = getPV_StreamFilepos(mTimestampFile);
         parent->writeScalarToFile(cpDir, getName(), "TimestampState", timestampFilePos);
      }
      return HyPerLayer::checkpointWrite(cpDir);
   }

   void BaseInput::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
      char *tempString;
      parent->ioParamStringRequired(ioFlag, name, "inputPath", &tempString); //TODO: These should really use std::string, we probably have tons of leaks
      mInputPath = std::string(tempString);
      free(tempString);

      // Check if the input path ends in ".txt" and enable the file list if so
      std::string txt = ".txt";
      if(mInputPath.compare(mInputPath.size() - txt.size(), txt.size(), txt) == 0) {
         mUsingFileList = true; //TODO: Add a flag to override this value even when the input path ends in ".txt"
      }
   }

   void BaseInput::ioParam_useInputBCflag(enum ParamsIOFlag ioFlag) { //TODO: Change to useInputBCFlag, add deprecated warning
      parent->ioParamValue(ioFlag, name, "useImageBCflag", &mUseInputBCflag, mUseInputBCflag);
   }

   int BaseInput::ioParam_offsets(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "offsetX", &mOffsets[0], mOffsets[0]);
      parent->ioParamValue(ioFlag, name, "offsetY", &mOffsets[1], mOffsets[1]);

      return PV_SUCCESS;
   }

   void BaseInput::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag){
      parent->ioParamString(ioFlag, name, "offsetAnchor", &mOffsetAnchor, "tl");
      if (ioFlag==PARAMS_IO_READ) {
         int status = checkValidAnchorString();
         if (status != PV_SUCCESS) {
            if (parent->columnId()==0) {
               pvErrorNoExit().printf("%s: offsetAnchor must be a two-letter string.  The first character must be \"t\", \"c\", or \"b\" (for top, center or bottom); and the second character must be \"l\", \"c\", or \"r\" (for left, center or right).\n", getDescription_c());
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
      }
   }

   void BaseInput::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "autoResizeFlag", &mAutoResizeFlag, mAutoResizeFlag);
   }

   void BaseInput::ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
      if (mAutoResizeFlag) {
         parent->ioParamString(ioFlag, name, "aspectRatioAdjustment", &mAspectRatioAdjustment, "crop"/*default*/);
         if (ioFlag == PARAMS_IO_READ) {
            // Check if the input path ends in ".txt" and enable the file list if so
            assert(mAspectRatioAdjustment);
            for (char * c = mAspectRatioAdjustment; *c; c++) { *c = tolower(*c); }
         }
         if (strcmp(mAspectRatioAdjustment, "crop") && strcmp(mAspectRatioAdjustment, "pad")) {
            if (parent->columnId()==0) {
               pvErrorNoExit().printf("%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                     getDescription_c());
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
      }
   }

   void BaseInput::ioParam_interpolationMethod(enum ParamsIOFlag ioFlag) {
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

   void BaseInput::ioParam_inverseFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "inverseFlag", &mInverseFlag, mInverseFlag);
   }

   void BaseInput::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "normalizeLuminanceFlag", &mNormalizeLuminanceFlag, mNormalizeLuminanceFlag);
   }

   void BaseInput::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeLuminanceFlag"));
      if (mNormalizeLuminanceFlag) {
        parent->ioParamValue(ioFlag, name, "normalizeStdDev", &mNormalizeStdDev, mNormalizeStdDev);
      }
   }
   void BaseInput::ioParam_padValue(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "padValue", &mPadValue, mPadValue);
   }

   void BaseInput::ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag) {
   //   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   //   if (mJitterFlag) {
         parent->ioParamValue(ioFlag, name, "offsetConstraintMethod", &mOffsetConstraintMethod, 0/*default*/);
         if (ioFlag == PARAMS_IO_READ && (mOffsetConstraintMethod <0 || mOffsetConstraintMethod >3) ) {
            pvError().printf("%s: offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getDescription_c());
         }
   //   }
   }

   void BaseInput::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
      assert(this->initVObject == NULL);
      return;
   }

   void BaseInput::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
      if (ioFlag == PARAMS_IO_READ) {
         triggerLayerName = NULL;
         triggerFlag = false;
         parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL/*correct value*/);
      }
   }

   int BaseInput::checkValidAnchorString() {
      int status = PV_SUCCESS;
      if (mOffsetAnchor==NULL || strlen(mOffsetAnchor) != (size_t) 2) {
         status = PV_FAILURE;
      }
      else {
         char xOffsetAnchor = mOffsetAnchor[1];
         if (xOffsetAnchor != 'l' && xOffsetAnchor != 'c' && xOffsetAnchor != 'r') {
            status = PV_FAILURE;
         }
         char yOffsetAnchor = mOffsetAnchor[0];
         if (yOffsetAnchor != 't' && yOffsetAnchor != 'c' && yOffsetAnchor != 'b') {
            status = PV_FAILURE;
         }
      }
      return status;
   }

   void BaseInput::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "displayPeriod", &mDisplayPeriod, mDisplayPeriod);
   }

   void BaseInput::ioParam_echoFramePathnameFlag(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "echoFramePathnameFlag", &mEchoFramePathnameFlag, false/*default value*/);
   }

   void BaseInput::ioParam_batchMethod(enum ParamsIOFlag ioFlag) {
      char *batchMethod;
      parent->ioParamString(ioFlag, name, "batchMethod", &batchMethod, "bySpecified");
      if(strcmp(batchMethod, "byImage") == 0 || strcmp(batchMethod, "byFile") == 0) {
         mBatchMethod = BYFILE;
      }
      else if(strcmp(batchMethod, "byMovie") == 0 || strcmp(batchMethod, "byList") == 0) {
         mBatchMethod = BYLIST;  
      } 
      else if(strcmp(batchMethod, "bySpecified") == 0) {
         mBatchMethod = BYSPECIFIED;
      }
      else{
         pvError() << "Input layer " << name << " batchMethod not recognized. Options are \"byFile\", \"byList\", and \"bySpecified\"\n";
      }
   }

   void BaseInput::ioParam_start_frame_index(enum ParamsIOFlag ioFlag) {
      int *paramsStartFrameIndex;
      int length = -1;
      this->getParent()->ioParamArray(ioFlag, this->getName(), "start_frame_index", &paramsStartFrameIndex, &length);
      mStartFrameIndex.clear();
      if(length > 0) {
         mStartFrameIndex.resize(length);
         for(int i = 0; i < length; ++i) {
            mStartFrameIndex.at(i) = paramsStartFrameIndex[i];
         }
      }
      free(paramsStartFrameIndex);
   }

   void BaseInput::ioParam_skip_frame_index(enum ParamsIOFlag ioFlag) {
      int *paramsSkipFrameIndex;
      int length = -1;
      this->getParent()->ioParamArray(ioFlag, this->getName(), "skip_frame_index", &paramsSkipFrameIndex, &length);
      mSkipFrameIndex.clear();
      if(length > 0) {
         mSkipFrameIndex.resize(length);
         for(int i = 0; i < length; ++i) {
            mSkipFrameIndex.at(i) = paramsSkipFrameIndex[i];
         }
      }
      free(paramsSkipFrameIndex);
   }

   void BaseInput::ioParam_writeFrameToTimestamp(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "writeFrameToTimestamp", &mWriteFileToTimestamp, mWriteFileToTimestamp);
   }

   void BaseInput::ioParam_resetToStartOnLoop(enum ParamsIOFlag ioFlag) {
      parent->ioParamValue(ioFlag, name, "resetToStartOnLoop", &mResetToStartOnLoop, mResetToStartOnLoop);
   }

   int BaseInput::allocateDataStructures() {

      //Allocate framePaths here before image, since allocate call will call getFrame
      int numBatch = parent->getNBatch();
 
      if(parent->getCommunicator()->commRank()==0){
         mFileList.resize(numBatch);
      }
      
      mFileIndices.resize(numBatch);
      
      //Calculate file positions for beginning of each frame
      populateFileList();
      pvInfo() << "File " << mInputPath << " contains " << mFileList.size() << " frames\n";

      mStartFrameIndex.resize(numBatch);
      mSkipFrameIndex.resize(numBatch);

      int numBatchGlobal = getLayerLoc()->nbatchGlobal;
      int kb0 = getLayerLoc()->kb0;
      int offset = 0;
      int framesPerBatch = 0;

      switch(mBatchMethod) {
         case BYFILE:
            //No skip here allowed
            pvErrorIf(mSkipFrameIndex.size() != 0, "%s: batchMethod of \"byFile\" sets skip_frame_index, do not specify.\n", getName());
            //Default value
            if(mStartFrameIndex.size() == 1) {
               offset = mStartFrameIndex.at(0);
            }
            pvErrorIf(mStartFrameIndex.size() > 1, "%s: batchMethod of \"byFile\" requires 0 or 1 start_frame_index values\n", getName());
            //Allocate and default
            for(int b = 0; b < numBatch; ++b) {
               mStartFrameIndex.at(b) = offset + kb0 + b;
               mSkipFrameIndex.at(b) = numBatchGlobal;
            }
            break;

         case BYLIST:
            pvErrorIf(mSkipFrameIndex.size() != 0, "%s: batchMethod of \"byImage\" sets skip_frame_index, do not specify.\n", getName());
            if(mStartFrameIndex.size() == 1) {
               offset = mStartFrameIndex.at(0);
            }
            pvErrorIf(mStartFrameIndex.size() > 1, "%s: batchMethod of \"byList\" requires 0 or 1 start_frame_index values\n", getName());
            framesPerBatch = floor(mFileList.size()/numBatchGlobal);
            if(framesPerBatch < 1) {
               framesPerBatch = 1;
            }
            for(int b = 0; b < numBatch; ++b) { 
               mStartFrameIndex.at(b) = offset + ((b+kb0)*framesPerBatch);
               mSkipFrameIndex.at(b) = 1;
            }
            break;

         case BYSPECIFIED:
            pvErrorIf(mStartFrameIndex.size() != numBatchGlobal && mStartFrameIndex.size() != 0,
               "%s: batchMethod of \"bySpecified\" requires 0 or %d start_frame_index values\n", getName(), numBatchGlobal);
            pvErrorIf(mSkipFrameIndex.size() != numBatchGlobal && mSkipFrameIndex.size() != 0,
               "%s batchMethod of \"bySpecified\" requires 0 or %d skip_frame_index values\n", getName(), numBatchGlobal);
            // Use default values if none were given, otherwise this was handleded when loading the params
            if(mStartFrameIndex.size() == 0) {
               mStartFrameIndex.push_back(0);
            }
            if(mSkipFrameIndex.size() == 0) {
               mSkipFrameIndex.push_back(1);
            }
            break;
      }

      if (parent->columnId() == 0) {
         for (int b = 0; b < numBatch; ++b) {
            mFileIndices.at(b) = -1;
         }
      }

      int status = HyPerLayer::allocateDataStructures();
      mInputData.resize(getLayerLoc()->ny, getLayerLoc()->nx, getLayerLoc()->nf);
      nextInput(parent->simulationTime(), parent->getDeltaTimeBase());
      // create mpi_datatypes for border transfer
      mDatatypes = Communicator::newDatatypes(getLayerLoc());
      exchange();
      return status;
   }


   double BaseInput::calcTimeScale(int batchIdx) {
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
   bool BaseInput::readyForNextFile() { 
      return true;
   }

   //TODO: This doesn't appear to be using mDisplayPeriod?
   int BaseInput::updateState(double time, double dt)
   {
      if(!mUsingFileList) {
         return PV_SUCCESS;
      }

      Communicator * icComm = getParent()->getCommunicator();
      //Only do this if it's not the first update timestep
      //The timestep number is (time - startTime)/(width of timestep), with allowance for roundoff.
      //But if we're using adaptive timesteps, the dt passed as a function argument is not the correct (width of timestep).
      if(fabs(time - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)) {
         if(readyForNextFile()) {
            nextInput(time, dt);
            //Write to timestamp file 
            if(icComm->commRank() == 0) {
               if(mTimestampFile) {
                  std::ostringstream outStrStream;
                  outStrStream.precision(15);
                  int kb0 = getLayerLoc()->kb0;
                  for(int b = 0; b < parent->getNBatch(); ++b) {
                     outStrStream << time << "," << b+kb0 << "," << mFileIndices.at(b) << "," << mFileList.at(b) << "\n";
                  }
                  size_t len = outStrStream.str().length();
                  int status = PV_fwrite(outStrStream.str().c_str(), sizeof(char), len, mTimestampFile) == len ? PV_SUCCESS : PV_FAILURE;
                  pvErrorIf(status != PV_SUCCESS, "%s: Movie::updateState failed to write to timestamp file.\n", getDescription_c());
                  fflush(mTimestampFile->fp);
               }
            }
         }
      } 
      return PV_SUCCESS;
   }

   //This function is only being called here from allocate. Subclasses will call this function when a new frame is nessessary
   void BaseInput::nextInput(double timef, double dt) {
      for(int b = 0; b < parent->getNBatch(); b++) {
         if (parent->columnId() == 0) {
            retrieveData(timef, dt, b);
         }
         scatterInput(b);
      }
      postProcess(timef, dt);
   }

   int BaseInput::scatterInput(int batchIndex) {
      const int rank = parent->columnId();
      const int rootProc = 0;
      MPI_Comm mpiComm = parent->getCommunicator()->communicator();
      pvadata_t *activityBuffer = getActivity() + batchIndex * getNumExtended();

      if (rank == rootProc) {
         const PVLayerLoc *loc = getLayerLoc();
         const PVHalo *halo = &loc->halo;
         const int numXExtended = loc->nx + halo->lt + halo->rt;
         const int numYExtended = loc->ny + halo->dn + halo->up;
         const int numFeatures = loc->nf;

         resizeInput();

         int dims[2] = { mInputData.getColumns(), mInputData.getRows() };
         std::vector<float> rawInput = mInputData.asVector();

         MPI_Bcast(dims, 2, MPI_INT, rootProc, mpiComm);
         
         for (int rank = 0; rank < parent->getCommunicator()->commSize(); ++rank) {
            if (rank == rootProc) {  // Do root process last so that we don't clobber root process data by using the data buffer to send.
               continue;
            }
            for (int n = 0; n < getNumExtended(); ++n) {
               activityBuffer[n] = mPadValue;
            }
            int layerLeft, layerTop, inputLeft, inputTop, width, height;
            int status = calcLocalBox(rank, layerLeft, layerTop, inputLeft, inputTop, width, height);
            if (status == PV_SUCCESS) {
               pvAssert(width > 0 && height > 0);
               for (int y = 0; y < height; ++y) {
                  int inputIdx = kIndex(inputLeft, inputTop + y, 0, mInputData.getColumns(), mInputData.getRows(), numFeatures);
                  int layerIdx = kIndex(layerLeft, layerTop + y, 0, numXExtended, numYExtended, numFeatures);
                  memcpy(&activityBuffer[layerIdx], &rawInput[inputIdx], sizeof(pvadata_t) * width * numFeatures);
               }
            }
            else {
               pvAssert(width == 0 || height == 0);
            }
            MPI_Send(activityBuffer, getNumExtended(), MPI_FLOAT, rank, 31, mpiComm);
         }
         // Finally, do root process.
         for (int n = 0; n < getNumExtended(); n++) {
            activityBuffer[n] = mPadValue;
         }
         int status = calcLocalBox(rootProc, mLayerLeft, mLayerTop, mInputLeft, mInputTop, mInputWidth, mInputHeight);
         if (status == PV_SUCCESS) {
            pvAssert(mInputWidth > 0 && mInputHeight > 0);
            for(int y = 0; y < mInputHeight; ++y) {
               int inputIdx = kIndex(mInputLeft, mInputTop + y, 0, mInputData.getColumns(), mInputData.getRows(), numFeatures);
               int layerIdx = kIndex(mLayerLeft, mLayerTop + y, 0, numXExtended, numYExtended, numFeatures);
               assert(inputIdx >= 0 && inputIdx < mInputData.getColumns() * mInputData.getRows() * numFeatures);
               assert(layerIdx >= 0 && layerIdx < getNumExtended());
               memcpy(&activityBuffer[layerIdx], &rawInput[inputIdx], sizeof(pvadata_t) * mInputWidth * numFeatures);
            }
         }
         else { assert(mInputWidth == 0 || mInputHeight == 0); }
      }
      else {
         int dims[2];
         MPI_Bcast(dims, 2, MPI_INT, rootProc, mpiComm);
         MPI_Recv(activityBuffer, getNumExtended(), MPI_FLOAT, rootProc, 31, mpiComm, MPI_STATUS_IGNORE);
         calcLocalBox(rank, mLayerLeft, mLayerTop, mInputLeft, mInputTop, mInputWidth, mInputHeight);
         mInputData.resize(dims[1], dims[0], getLayerLoc()->nf); 
      }
      return PV_SUCCESS;
   }

   int BaseInput::resizeInput() {
      pvAssert(parent->columnId() == 0); // Should only be called by root process.

      if (!mAutoResizeFlag) {
         return PV_SUCCESS;
      }
      
      const PVLayerLoc *loc = getLayerLoc();
      const PVHalo *halo = &loc->halo;
      const int targetWidth = loc->nxGlobal + (mUseInputBCflag ? (halo->lt + halo->rt) : 0);
      const int targetHeight = loc->nyGlobal + (mUseInputBCflag ? (halo->dn + halo->up) : 0);

      Buffer::RescaleMethod rescaleMethod = Buffer::CROP;
      if(!strcmp(mAspectRatioAdjustment, "pad")) {
         rescaleMethod = Buffer::PAD;
      }

      mInputData.rescale(targetHeight, targetWidth, rescaleMethod, mInterpolationMethod);
      
      return PV_SUCCESS;
   }

   int BaseInput::calcLocalBox(int rank, int &layerLeft, int &layerTop, int &inputLeft, int &inputTop, int &width, int &height) {
      Communicator *icComm = parent->getCommunicator();
      const PVLayerLoc *loc = getLayerLoc();
      const PVHalo *halo = &loc->halo;
      int column = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
      int boxInInputLeft = getOffsetX(mOffsetAnchor, mOffsets[0]) + column * getLayerLoc()->nx;
      int boxInInputRight = boxInInputLeft + loc->nx;
      int boxInLayerLeft = halo->lt;
      int boxInLayerRight = boxInLayerLeft + loc->nx;
      int row = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
      int boxInInputTop = getOffsetY(mOffsetAnchor, mOffsets[1]) + row * getLayerLoc()->ny;
      int boxInInputBottom = boxInInputTop + loc->ny;
      int boxInLayerTop = halo->up;
      int boxInLayerBottom = boxInLayerTop + loc->ny;

      if (mUseInputBCflag) {
         boxInLayerLeft -= halo->lt;
         boxInLayerRight += halo->rt;
         boxInInputLeft -= halo->lt;
         boxInInputRight += halo->rt;

         boxInLayerTop -= halo->up;
         boxInLayerBottom += halo->dn;
         boxInInputTop -= halo->up;
         boxInInputBottom += halo->dn;
      }

      int status = PV_SUCCESS;
      if (boxInInputLeft > mInputData.getColumns() || boxInInputRight < 0 ||
          boxInInputTop > mInputData.getRows() || boxInInputBottom < 0 ||
          boxInLayerLeft > loc->nx+halo->lt+halo->rt || boxInLayerRight < 0 ||
          boxInLayerTop > loc->ny+halo->dn+halo->up || boxInLayerBottom < 0) {
         width = 0;
         height = 0;
         status = PV_FAILURE;
      }
      else {
         if (boxInInputLeft < 0) {
            int discrepancy = -boxInInputLeft;
            boxInLayerLeft += discrepancy;
            boxInInputLeft += discrepancy;
         }
         if (boxInLayerLeft < 0) {
            int discrepancy = -boxInLayerLeft;
            boxInLayerLeft += discrepancy;
            boxInInputLeft += discrepancy;
         }

         if (boxInInputRight > mInputData.getColumns()) {
            int discrepancy = boxInInputRight - mInputData.getColumns();
            boxInLayerRight -= discrepancy;
            boxInInputRight -= discrepancy;
         }
         if (boxInLayerRight > loc->nx+halo->lt+halo->rt) {
            int discrepancy = boxInLayerRight - (loc->nx+halo->lt+halo->rt);
            boxInLayerRight -= discrepancy;
            boxInInputRight -= discrepancy;
         }

         if (boxInInputTop < 0) {
            int discrepancy = -boxInInputTop;
            boxInLayerTop += discrepancy;
            boxInInputTop += discrepancy;
         }
         if (boxInLayerTop < 0) {
            int discrepancy = -boxInLayerTop;
            boxInLayerTop += discrepancy;
            boxInInputTop += discrepancy;
         }

         if (boxInInputBottom > mInputData.getRows()) {
            int discrepancy = boxInInputBottom - mInputData.getRows();
            boxInLayerBottom -= discrepancy;
            boxInInputBottom -= discrepancy;
         }
         if (boxInLayerBottom > loc->ny+halo->dn+halo->up) {
            int discrepancy = boxInLayerRight - (loc->ny+halo->dn+halo->up);
            boxInLayerBottom -= discrepancy;
            boxInInputBottom -= discrepancy;
         }

         int boxWidth = boxInInputRight-boxInInputLeft;
         int boxHeight = boxInInputBottom-boxInInputTop;
         pvAssert(boxWidth >= 0);
         pvAssert(boxHeight >= 0);
         pvAssert(boxWidth == boxInLayerRight - boxInLayerLeft);
         pvAssert(boxHeight == boxInLayerBottom - boxInLayerTop);
         // coordinates can be outside of the imageLoc limits, as long as the box has zero area
         pvAssert(boxInInputLeft >= 0 && boxInInputRight <= mInputData.getColumns());
         pvAssert(boxInInputTop >= 0 && boxInInputBottom <= mInputData.getRows());
         pvAssert(boxInLayerLeft >= 0 && boxInLayerRight <= loc->nx+halo->lt+halo->rt);
         pvAssert(boxInLayerTop >= 0 && boxInLayerBottom <= loc->ny+halo->dn+halo->up);

         layerLeft = boxInLayerLeft;
         layerTop = boxInLayerTop;
         inputLeft = boxInInputLeft;
         inputTop = boxInInputTop;
         width = boxWidth;
         height = boxHeight;
      }
      return status;
   }

   //Apply normalizeLuminanceFlag, normalizeStdDev, and inverseFlag, which can be done pixel-by-pixel
   //after scattering.
   int BaseInput::postProcess(double timef, double dt){
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


   //Offsets based on an anchor point, so calculate offsets based off a given anchor point
   //Note: imageLoc must be updated before calling this function
   int BaseInput::getOffsetX(const char* offsetAnchor, int offsetX){
      if(!strcmp(offsetAnchor, "tl") || !strcmp(offsetAnchor, "cl") || !strcmp(offsetAnchor, "bl")){
         return offsetX;
      }
      //Offset in center
      else if(!strcmp(offsetAnchor, "tc") || !strcmp(offsetAnchor, "cc") || !strcmp(offsetAnchor, "bc")){
         int layerSizeX = getLayerLoc()->nxGlobal;
         return ((mInputData.getColumns()/2)-(layerSizeX/2)) + offsetX;
      }
      //Offset on bottom
      else if(!strcmp(offsetAnchor, "tr") || !strcmp(offsetAnchor, "cr") || !strcmp(offsetAnchor, "br")){
         int layerSizeX = getLayerLoc()->nxGlobal;
         return (mInputData.getColumns() - layerSizeX) + offsetX;
      }
      assert(0); // All possible cases should be covered above
      return -1; // Eliminates no-return warning
   }

   int BaseInput::getOffsetY(const char* offsetAnchor, int offsetY){
      //Offset on top
      if(!strcmp(offsetAnchor, "tl") || !strcmp(offsetAnchor, "tc") || !strcmp(offsetAnchor, "tr")){
         return offsetY;
      }
      //Offset in center
      else if(!strcmp(offsetAnchor, "cl") || !strcmp(offsetAnchor, "cc") || !strcmp(offsetAnchor, "cr")){
         int layerSizeY = getLayerLoc()->nyGlobal;
         return ((mInputData.getRows()/2)-(layerSizeY/2)) + offsetY;
      }
      //Offset on bottom
      else if(!strcmp(offsetAnchor, "bl") || !strcmp(offsetAnchor, "bc") || !strcmp(offsetAnchor, "br")){
         int layerSizeY = getLayerLoc()->nyGlobal;
         return (mInputData.getRows()-layerSizeY) + offsetY;
      }
      assert(0); // All possible cases should be covered above
      return -1; // Eliminates no-return warning
   }


   bool BaseInput::constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method) {
      bool moved_x = point[0] < min_x || point[0] > max_x;
      bool moved_y = point[1] < min_y || point[1] > max_y;
      if (moved_x) {
         if (min_x > max_x) {
            pvError().printf("Image::constrainPoint error.  min_x=%d is greater than max_x= %d\n", min_x, max_x);
         }
         int size_x = max_x-min_x;
         int new_x = point[0];
         switch (method) {
         case 0: // Ignore
            break;
         case 1: // Mirror
            new_x -= min_x;
            new_x %= (2*(size_x+1));
            if (new_x<0) new_x++;
            new_x = abs(new_x);
            if (new_x>size_x) new_x = 2*size_x+1-new_x;
            new_x += min_x;
            break;
         case 2: // Stick to wall
            if (new_x<min_x) new_x = min_x;
            if (new_x>max_x) new_x = max_x;
            break;
         case 3: // Circular
            new_x -= min_x;
            new_x %= size_x+1;
            if (new_x<0) new_x += size_x+1;
            new_x += min_x;
            break;
         default:
            pvAssertMessage(0, "Method type \"%d\" not understood\n", method);
            break;
         }
         assert(new_x >= min_x && new_x <= max_x);
         point[0] = new_x;
      }
      if (moved_y) {
         if (min_y > max_y) {
            pvError().printf("Image::constrainPoint error.  min_y=%d is greater than max_y=%d\n", min_y, max_y);
         }
         int size_y = max_y-min_y;
         int new_y = point[1];
         switch (method) {
         case 0: // Ignore
            break;
         case 1: // Mirror
            new_y -= min_y;
            new_y %= (2*(size_y+1));
            if (new_y<0) new_y++;
            new_y = abs(new_y);
            if (new_y>size_y) new_y = 2*size_y+1-new_y;
            new_y += min_y;
            break;
         case 2: // Stick to wall
            if (new_y<min_y) new_y = min_y;
            if (new_y>max_y) new_y = max_y;
            break;
         case 3: // Circular
            new_y -= min_y;
            new_y %= size_y+1;
            if (new_y<0) new_y += size_y+1;
            new_y += min_y;
            break;
         default:
            assert(0);
            break;
         }
         assert(new_y >= min_y && new_y <= max_y);
         point[1] = new_y;
      }
      return moved_x || moved_y;
   }
   bool BaseInput::constrainOffsets() {
      int newOffsets[2];
      int oldOffsetX = getOffsetX(mOffsetAnchor, mOffsets[0]);
      int oldOffsetY = getOffsetY(mOffsetAnchor, mOffsets[1]);
      newOffsets[0] = oldOffsetX; 
      newOffsets[1] = oldOffsetY; 
      bool status = constrainPoint(newOffsets, 0, mInputData.getColumns() - getLayerLoc()->nxGlobal, 0, mInputData.getRows() - getLayerLoc()->nyGlobal, mOffsetConstraintMethod);
      int diffx = newOffsets[0] - oldOffsetX;
      int diffy = newOffsets[1] - oldOffsetY;
      mOffsets[0] = mOffsets[0] + diffx;
      mOffsets[1] = mOffsets[1] + diffy;
      return status;
   }

   void BaseInput::exchange() {
      std::vector<MPI_Request> req{};
      for (int b=0; b<getLayerLoc()->nbatch; ++b) {
         parent->getCommunicator()->exchange(getActivity()+b*getNumExtended(), mDatatypes, getLayerLoc(), req);
         parent->getCommunicator()->wait(req);
         pvAssert(req.empty());
      }
   }

   int BaseInput::requireChannel(int channelNeeded, int * numChannelsResult) {
      if (parent->columnId()==0) {
         pvErrorNoExit().printf("%s cannot be a post-synaptic layer.\n",
               getDescription_c());
      }
      *numChannelsResult = 0;
      return PV_FAILURE;
   }

   int BaseInput::allocateV() {
      clayer->V = NULL;
      return PV_SUCCESS;
   }

   int BaseInput::initializeV() {
      assert(getV()==NULL);
      return PV_SUCCESS;
   }

   int BaseInput::initializeActivity() {
      return PV_SUCCESS;
   }

   // advance by n_skip lines through file of filenames, always advancing at least one line
   std::string BaseInput::getNextFilename(int filesToSkip, int batchIdx) {
      Communicator * icComm = getParent()->getCommunicator();
      pvAssert(icComm->commRank() == 0);
      std::string outFilename;
      if(filesToSkip < 1) {
         filesToSkip = 1;
      }
      for (int skipIndex = 0; skipIndex < filesToSkip; ++skipIndex) {
         outFilename = advanceFilename(batchIdx);
      }
      if (mEchoFramePathnameFlag) {
         pvInfo().printf("%s: t=%f, batch element %d: loading %s\n", getDescription_c(), parent->simulationTime(), batchIdx, outFilename.c_str());
      }
      return outFilename;
   }

   //This function will reset the file position of the open file
   int BaseInput::populateFileList() {
      int count = 0;
      if(parent->columnId() == 0) {
         std::string line;
         mFileList.clear();
         std::ifstream infile(mInputPath, std::ios_base::in);
         while(getline(infile, line, '\n')) {
            std::string noWhiteSpace = line;
            noWhiteSpace.erase(std::remove_if(noWhiteSpace.begin(), noWhiteSpace.end(), ::isspace), noWhiteSpace.end());
            if(!noWhiteSpace.empty()) {
               mFileList.push_back(line);
            }
         }
         mFileIndices.at(0) = -1;;
         count = mFileList.size();
      }
      
      MPI_Bcast(&count, 1, MPI_INT, 0, parent->getCommunicator()->communicator());

      // Maintain the file count on child processes
      if(parent->columnId() != 0) {
         mFileList.clear();
         mFileList.resize(count);
      }
      return count;
   }

   std::string BaseInput::advanceFilename(int batchIndex) {
      pvAssert(parent->columnId() == 0);
      if(++mFileIndices.at(batchIndex) >= mFileList.size()) {
         pvInfo() << getName() << ": End of file list reached. Rewinding." << std::endl;
         if(mResetToStartOnLoop) {
            mFileIndices.at(batchIndex) = mStartFrameIndex.at(batchIndex);
         }
         else {
            mFileIndices.at(batchIndex) = 0;
         }
      }
      return mFileList.at(mFileIndices.at(batchIndex));
   }
} 





