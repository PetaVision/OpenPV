/*
 * BaseInput.cpp
 */

#include "BaseInput.hpp"

namespace PV {

BaseInput::BaseInput() {
   initialize_base();
}

BaseInput::~BaseInput() {
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   delete randState; randState = NULL;

   free(aspectRatioAdjustment);

   if(writePosition){
      if (getParent()->icCommunicator()->commRank()==0 && fp_pos != NULL && fp_pos->isfile) {
         PV_fclose(fp_pos);
      }
   }
   if(offsetAnchor){
      free(offsetAnchor);
   }
   free(writeImagesExtension);
   free(inputPath);
   delete[] imageData;
}

int BaseInput::initialize_base() {
   numChannels = 0;
   mpi_datatypes = NULL;
   data = NULL;
   imageData = NULL;
   memset(&imageLoc, 0, sizeof(PVLayerLoc));
   imageColorType = COLORTYPE_UNRECOGNIZED;
   useImageBCflag = false;
   autoResizeFlag = false;
   aspectRatioAdjustment = NULL;
   interpolationMethod = INTERPOLATE_UNDEFINED;
   writeImages = false;
   writeImagesExtension = NULL;
   inverseFlag = false;
   normalizeLuminanceFlag = false;
   normalizeStdDev = true;
   offsets[0] = 0;
   offsets[1] = 0;
   offsetAnchor = NULL;
   jitterFlag = false;
   jitterType = RANDOM_WALK;
   timeSinceLastJitter = 0;
   jitterRefractoryPeriod = 0;
   stepSize = 0;
   persistenceProb = 0.0;
   recurrenceProb = 1.0;
   biasChangeTime = FLT_MAX;
   writePosition = 0;
   fp_pos = NULL;
   biases[0]   = 0;
   biases[1]   = 0;
   randState = NULL;
   biasConstraintMethod = 0; 
   padValue = 0;
   inputPath = NULL;
   return PV_SUCCESS;
}

int BaseInput::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);

   this->lastUpdateTime = parent->getStartTime();

   PVParams * params = parent->parameters();

   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      assert(!params->presentAndNotBeenRead(name, "offsetX"));
      assert(!params->presentAndNotBeenRead(name, "offsetY"));
      assert(!params->presentAndNotBeenRead(name, "offsetAnchor"));
      biases[0] = getOffsetX(this->offsetAnchor, offsets[0]);
      biases[1] = getOffsetY(this->offsetAnchor, offsets[1]);
   }

   return status;
}

int BaseInput::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);

   ioParam_inputPath(ioFlag);
   ioParam_offsetAnchor(ioFlag);
   ioParam_offsets(ioFlag);
   ioParam_writeImages(ioFlag);
   ioParam_writeImagesExtension(ioFlag);

   ioParam_autoResizeFlag(ioFlag);
   ioParam_aspectRatioAdjustment(ioFlag);
   ioParam_interpolationMethod(ioFlag);
   ioParam_inverseFlag(ioFlag);
   ioParam_normalizeLuminanceFlag(ioFlag);
   ioParam_normalizeStdDev(ioFlag);

   ioParam_jitterFlag(ioFlag);
   ioParam_jitterType(ioFlag);
   ioParam_jitterRefractoryPeriod(ioFlag);
   ioParam_stepSize(ioFlag);
   ioParam_persistenceProb(ioFlag);
   ioParam_recurrenceProb(ioFlag);
   ioParam_biasChangeTime(ioFlag);
   ioParam_biasConstraintMethod(ioFlag);
   ioParam_offsetConstraintMethod(ioFlag);
   ioParam_writePosition(ioFlag);
   //ioParam_useParamsImage(ioFlag);
   ioParam_useImageBCflag(ioFlag);

   ioParam_padValue(ioFlag);

   return status;
}

void BaseInput::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputPath", &inputPath);
}

void BaseInput::ioParam_useImageBCflag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "useImageBCflag", &useImageBCflag, useImageBCflag);
}

int BaseInput::ioParam_offsets(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "offsetX", &offsets[0], offsets[0]);
   parent->ioParamValue(ioFlag, name, "offsetY", &offsets[1], offsets[1]);

   return PV_SUCCESS;
}

void BaseInput::ioParam_offsetAnchor(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "offsetAnchor", &offsetAnchor, "tl");
   if (ioFlag==PARAMS_IO_READ) {
      int status = checkValidAnchorString();
      if (status != PV_SUCCESS) {
         if (parent->columnId()==0) {
            pvErrorNoExit().printf("%s: offsetAnchor must be a two-letter string.  The first character must be \"t\", \"c\", or \"b\" (for top, center or bottom); and the second character must be \"l\", \"c\", or \"r\" (for left, center or right).\n", getDescription_c());
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void BaseInput::ioParam_writeImages(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeImages", &writeImages, writeImages);
}

void BaseInput::ioParam_writeImagesExtension(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeImages"));
   if (writeImages) {
      parent->ioParamString(ioFlag, name, "writeImagesExtension", &writeImagesExtension, "tif");
   }
}

void BaseInput::ioParam_autoResizeFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "autoResizeFlag", &autoResizeFlag, autoResizeFlag);
}

void BaseInput::ioParam_aspectRatioAdjustment(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
   if (autoResizeFlag) {
      parent->ioParamString(ioFlag, name, "aspectRatioAdjustment", &aspectRatioAdjustment, "crop"/*default*/);
      if (ioFlag == PARAMS_IO_READ) {
         assert(aspectRatioAdjustment);
         for (char * c = aspectRatioAdjustment; *c; c++) { *c = tolower(*c); }
      }
      if (strcmp(aspectRatioAdjustment, "crop") && strcmp(aspectRatioAdjustment, "pad")) {
         if (parent->columnId()==0) {
            pvErrorNoExit().printf("%s: aspectRatioAdjustment must be either \"crop\" or \"pad\".\n",
                  getDescription_c());
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void BaseInput::ioParam_interpolationMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "autoResizeFlag"));
   if (autoResizeFlag) {
      char * interpolationMethodString = NULL;
      if (ioFlag == PARAMS_IO_READ) {
         parent->ioParamString(ioFlag, name, "interpolationMethod", &interpolationMethodString, "bicubic", true/*warn if absent*/);
         assert(interpolationMethodString);
         for (char * c = interpolationMethodString; *c; c++) { *c = tolower(*c); }
         if (!strncmp(interpolationMethodString, "bicubic", strlen("bicubic"))) {
            interpolationMethod = INTERPOLATE_BICUBIC;
         }
         else if (!strncmp(interpolationMethodString, "nearestneighbor", strlen("nearestneighbor"))) {
            interpolationMethod = INTERPOLATE_NEARESTNEIGHBOR;
         }
         else {
            if (parent->columnId()==0) {
               pvErrorNoExit().printf("%s: interpolationMethod must be either \"bicubic\" or \"nearestNeighbor\".\n",
                     getDescription_c());
            }
            MPI_Barrier(parent->icCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
      }
      else {
         assert(ioFlag == PARAMS_IO_WRITE);
         switch (interpolationMethod) {
         case INTERPOLATE_BICUBIC:
            interpolationMethodString = strdup("bicubic");
            break;
         case INTERPOLATE_NEARESTNEIGHBOR:
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
   parent->ioParamValue(ioFlag, name, "inverseFlag", &inverseFlag, inverseFlag);
}

void BaseInput::ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "normalizeLuminanceFlag", &normalizeLuminanceFlag, normalizeLuminanceFlag);
}

void BaseInput::ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "normalizeLuminanceFlag"));
   if (normalizeLuminanceFlag) {
     parent->ioParamValue(ioFlag, name, "normalizeStdDev", &normalizeStdDev, normalizeStdDev);
   }
}

void BaseInput::ioParam_jitterFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "jitterFlag", &jitterFlag, jitterFlag);
}

void BaseInput::ioParam_jitterType(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "jitterType", &jitterType, jitterType);
   }
}

void BaseInput::ioParam_jitterRefractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "jitterRefractoryPeriod", &jitterRefractoryPeriod, jitterRefractoryPeriod);
   }
}

void BaseInput::ioParam_stepSize(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "stepSize", &stepSize, stepSize);
   }
}

void BaseInput::ioParam_persistenceProb(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "persistenceProb", &persistenceProb, persistenceProb);
   }
}

void BaseInput::ioParam_recurrenceProb(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "recurrenceProb", &recurrenceProb, recurrenceProb);
   }
}

void BaseInput::ioParam_padValue(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "padValue", &padValue, padValue);
}

void BaseInput::ioParam_biasChangeTime(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "biasChangeTime", &biasChangeTime, biasChangeTime);
      if (ioFlag == PARAMS_IO_READ) {
         if (biasChangeTime < 0) {
            biasChangeTime = FLT_MAX;
         }
         nextBiasChange = parent->getStartTime() + biasChangeTime;
      }
   }
}

void BaseInput::ioParam_biasConstraintMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "biasConstraintMethod", &biasConstraintMethod, biasConstraintMethod);
      if (ioFlag == PARAMS_IO_READ && (biasConstraintMethod <0 || biasConstraintMethod >3)) {
         pvError().printf("%s: biasConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n",
               getDescription_c());
      }
   }
}

void BaseInput::ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "offsetConstraintMethod", &offsetConstraintMethod, 0/*default*/);
      if (ioFlag == PARAMS_IO_READ && (offsetConstraintMethod <0 || offsetConstraintMethod >3) ) {
         pvError().printf("%s: offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getDescription_c());
      }
   }
}

void BaseInput::ioParam_writePosition(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "writePosition", &writePosition, writePosition);
   }
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
   if (offsetAnchor==NULL || strlen(offsetAnchor) != (size_t) 2) {
      status = PV_FAILURE;
   }
   else {
      char xOffsetAnchor = offsetAnchor[1];
      if (xOffsetAnchor != 'l' && xOffsetAnchor != 'c' && xOffsetAnchor != 'r') {
         status = PV_FAILURE;
      }
      char yOffsetAnchor = offsetAnchor[0];
      if (yOffsetAnchor != 't' && yOffsetAnchor != 'c' && yOffsetAnchor != 'b') {
         status = PV_FAILURE;
      }
   }
   return status;
}

int BaseInput::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   if (jitterFlag) {
      status = initRandState();
   }

   data = clayer->activity->data;

   status = getFrame(parent->simulationTime(), parent->getDeltaTimeBase());
   assert(status == PV_SUCCESS);

   // readImage sets imageLoc based on the indicated file.  If filename is null, imageLoc doesn't change.

   // Open the file recording jitter positions.
   // This is in allocateDataStructures in case a subclass does something weird with the offsets, causing
   // the initial offsets to be unknown until the allocateDataStructures phase
   if(jitterFlag && writePosition){
      // Note: biasX and biasY are used only to calculate offsetX and offsetY;
      //       offsetX and offsetY are used only by readImage;
      //       readImage only uses the offsets in the zero-rank process
      // Therefore, the other ranks do not need to have their offsets stored.
      // In fact, it would be reasonable for the nonzero ranks not to compute biases and offsets at all,
      // but I chose not to fill the code with even more if(rank==0) statements.
      if( parent->icCommunicator()->commRank()==0 ) {
         char file_name[PV_PATH_MAX];

         int nchars = snprintf(file_name, PV_PATH_MAX, "%s/%s_jitter.txt", parent->getOutputPath(), getName());
         if (nchars >= PV_PATH_MAX) {
            pvError().printf("Path for jitter positions \"%s/%s_jitter.txt is too long.\n", parent->getOutputPath(), getName());
         }
         pvInfo().printf("%s will write jitter positions to %s\n", getDescription_c(), file_name);
         fp_pos = PV_fopen(file_name,"w",parent->getVerifyWrites());
         if(fp_pos == NULL) {
            pvError().printf("%s unable to open file \"%s\" for writing jitter positions.\n", getDescription_c(), file_name);
         }
         fprintf(fp_pos->fp,"%s, t=%f, bias x=%d y=%d, offset x=%d y=%d\n",getDescription_c(),parent->simulationTime(),biases[0],biases[1],
               getOffsetX(this->offsetAnchor, this->offsets[0]),getOffsetY(this->offsetAnchor, this->offsets[1]));
      }
   }

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

   // exchange border information
   exchange();

   return status;
}


//This function is only being called here from allocate. Subclasses will call this function when a new frame is nessessary
int BaseInput::getFrame(double timef, double dt) {
   int status = PV_SUCCESS;
   for(int b = 0; b < parent->getNBatch(); b++) {
      if (parent->columnId()==0) {
         if (status == PV_SUCCESS) { status = retrieveData(timef, dt, b); }
      }
      if (status == PV_SUCCESS) { status = scatterInput(b); }
   }
   if (status == PV_SUCCESS) { status = postProcess(timef, dt); }
   return status;
}

int BaseInput::scatterInput(int batchIndex) {
   int const rank = parent->columnId();
   MPI_Comm mpi_comm = parent->icCommunicator()->communicator();
   int const rootproc = 0;
   pvadata_t * A = data + batchIndex * getNumExtended();
   if (rank == rootproc) {
      PVLayerLoc const * loc = getLayerLoc();
      PVHalo const * halo = &loc->halo;
      int const nxExt = loc->nx + halo->lt + halo->rt;
      int const nyExt = loc->ny + halo->dn + halo->up;
      int const nf = loc->nf;

      if (imageLoc.nf==1 && nf>1) {
         convertGrayScaleToMultiBand(&imageData, imageLoc.nxGlobal, imageLoc.nyGlobal, nf);
         imageLoc.nf = nf;
         imageColorType = COLORTYPE_UNRECOGNIZED;
      }
      if (imageLoc.nf>1 && nf==1) {
         convertToGrayScale(&imageData, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf, imageColorType);
         imageLoc.nf = 1;
         imageColorType = COLORTYPE_GRAYSCALE;
      }
      if (imageLoc.nf != nf) {
         pvError().printf("%s: imageLoc has %d features but layer has %d features.\n",
               getDescription_c(), imageLoc.nf, nf);
      }

      resizeInput();
      int dims[2] = {imageLoc.nxGlobal, imageLoc.nyGlobal};
      MPI_Bcast(dims, 2, MPI_INT, rootproc, mpi_comm);
      for (int r=0; r<parent->icCommunicator()->commSize(); r++) {
         if (r==rootproc) { continue; } // Do root process last so that we don't clobber root process data by using the data buffer to send.
         for (int n=0; n<getNumExtended(); n++) { A[n] = padValue; }
         int dataLeft, dataTop, imageLeft, imageTop, width, height;
         int status = calcLocalBox(r, &dataLeft, &dataTop, &imageLeft, &imageTop, &width, &height);
         if (status==PV_SUCCESS) {
            pvAssert(width>0 && height>0);
            for (int y=0; y<height; y++) {
               int imageIdx = kIndex(imageLeft, imageTop+y, 0, imageLoc.nxGlobal, imageLoc.nyGlobal, nf);
               int dataIdx = kIndex(dataLeft, dataTop+y, 0, nxExt, nyExt, nf);
               memcpy(&A[dataIdx], &imageData[imageIdx], sizeof(pvadata_t)*width*nf);
            }
         }
         else {
            pvAssert(width==0||height==0);
         }
         MPI_Send(A, getNumExtended(), MPI_FLOAT, r, 31, mpi_comm);
      }
      // Finally, do root process.
      for (int n=0; n<getNumExtended(); n++) { A[n] = padValue; }
      int status = calcLocalBox(rootproc, &dataLeft, &dataTop, &imageLeft, &imageTop, &dataWidth, &dataHeight);
      if (status==PV_SUCCESS) {
         pvAssert(dataWidth>0 && dataHeight>0);
         for (int y=0; y<dataHeight; y++) {
            int imageIdx = kIndex(imageLeft, imageTop+y, 0, imageLoc.nxGlobal, imageLoc.nyGlobal, nf);
            int dataIdx = kIndex(dataLeft, dataTop+y, 0, nxExt, nyExt, nf);
            assert(imageIdx>=0 && imageIdx<imageLoc.nxGlobal * imageLoc.nyGlobal * imageLoc.nf);
            assert(dataIdx>=0 && dataIdx<getNumExtended());
            memcpy(&A[dataIdx], &imageData[imageIdx], sizeof(pvadata_t)*dataWidth*nf);
         }
      }
      else { assert(dataWidth==0||dataHeight==0); }
   }
   else {
      int dims[2];
      MPI_Bcast(dims, 2, MPI_INT, rootproc, mpi_comm);
      imageLoc.nxGlobal = dims[0];
      imageLoc.nyGlobal = dims[1];
      imageLoc.nf = getLayerLoc()->nf;
      MPI_Recv(A, getNumExtended(), MPI_FLOAT, rootproc, 31, mpi_comm, MPI_STATUS_IGNORE);
      calcLocalBox(rank, &dataLeft, &dataTop, &imageLeft, &imageTop, &dataWidth, &dataHeight);
   }
   return PV_SUCCESS;
}

int BaseInput::resizeInput() {
   pvAssert(parent->columnId()==0); // Should only be called by root process.
   if (!autoResizeFlag) {
      resizeFactor = 1.0f;
      return PV_SUCCESS;
   }
   PVLayerLoc const * loc = getLayerLoc();
   PVHalo const * halo = &loc->halo;
   int const targetWidth = loc->nxGlobal + (useImageBCflag ? (halo->lt + halo->rt) : 0);
   int const targetHeight = loc->nyGlobal + (useImageBCflag ? (halo->dn + halo->up) : 0);
   float xRatio = (float) targetWidth/(float) imageLoc.nxGlobal;
   float yRatio = (float) targetHeight/(float) imageLoc.nyGlobal;
   int resizedWidth, resizedHeight;
   if (!strcmp(aspectRatioAdjustment, "crop")) {
      resizeFactor = xRatio < yRatio ? yRatio : xRatio;
      // resizeFactor * width should be >= getLayerLoc()->nx; resizeFactor * height should be >= getLayerLoc()->ny,
      // and one of these relations should be == (up to floating-point roundoff).
      resizedWidth = (int) nearbyintf(resizeFactor * imageLoc.nxGlobal);
      resizedHeight = (int) nearbyintf(resizeFactor * imageLoc.nyGlobal);
      pvAssert(resizedWidth >= targetWidth);
      pvAssert(resizedHeight >= targetHeight);
      pvAssert(resizedWidth == targetWidth || resizedHeight == targetHeight);
   }
   else if (!strcmp(aspectRatioAdjustment, "pad")) {
      resizeFactor = xRatio < yRatio ? xRatio : yRatio;
      // resizeFactor * width should be <= getLayerLoc()->nx; resizeFactor * height should be <= getLayerLoc()->ny,
      // and one of these relations should be == (up to floating-point roundoff).
      resizedWidth = (int) nearbyintf(resizeFactor * imageLoc.nxGlobal);
      resizedHeight = (int) nearbyintf(resizeFactor * imageLoc.nyGlobal);
      pvAssert(resizedWidth <= getLayerLoc()->nxGlobal);
      pvAssert(resizedHeight <= getLayerLoc()->nyGlobal);
      pvAssert(resizedWidth == getLayerLoc()->nxGlobal || resizedHeight == getLayerLoc()->nxGlobal);
   }
   else {
      assert(0);
   }
   float * newImageData = new float[resizedHeight*resizedWidth*imageLoc.nf];
   switch(interpolationMethod) {
   case INTERPOLATE_BICUBIC:
      bicubicInterp(imageData, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf, imageLoc.nf, imageLoc.nf*imageLoc.nxGlobal, 1, newImageData, resizedWidth, resizedHeight);
      break;
   case INTERPOLATE_NEARESTNEIGHBOR:
      nearestNeighborInterp(imageData, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf, imageLoc.nf, imageLoc.nf*imageLoc.nxGlobal, 1, newImageData, resizedWidth, resizedHeight);
      break;
   default:
      assert(0);
      break;
   }
   delete[] imageData;
   imageData = newImageData;
   imageLoc.nxGlobal = resizedWidth;
   imageLoc.nyGlobal = resizedHeight;
   return PV_SUCCESS;
}
int BaseInput::nearestNeighborInterp(pvadata_t const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, pvadata_t * bufferOut, int widthOut, int heightOut) {
   /* Interpolation using nearest neighbor interpolation */
   int xinteger[widthOut];
   float dx = (float) (widthIn-1)/(float) (widthOut-1);
   for (int kx=0; kx<widthOut; kx++) {
      float x = dx * (float) kx;
      xinteger[kx] = (int) nearbyintf(x);
   }

   int yinteger[heightOut];
   float dy = (float) (heightIn-1)/(float) (heightOut-1);
   for (int ky=0; ky<heightOut; ky++) {
      float y = dy * (float) ky;
      yinteger[ky] = (int) nearbyintf(y);
   }

   for (int ky=0; ky<heightOut; ky++) {
      float yfetch = yinteger[ky];
      for (int kx=0; kx<widthOut; kx++) {
         int xfetch = xinteger[kx];
         for (int f=0; f<numBands; f++) {
            int fetchIdx = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
            int outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
            bufferOut[outputIdx] = bufferIn[fetchIdx];
         }
      }
   }
   return PV_SUCCESS;
}

int BaseInput::bicubicInterp(pvadata_t const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, pvadata_t * bufferOut, int widthOut, int heightOut) {
   /* Interpolation using bicubic convolution with a=-1 (following Octave image toolbox's imremap function -- change this?). */
   float xinterp[widthOut];
   int xinteger[widthOut];
   float xfrac[widthOut];
   float dx = (float) (widthIn-1)/(float) (widthOut-1);
   for (int kx=0; kx<widthOut; kx++) {
      float x = dx * (float) kx;
      xinterp[kx] = x;
      float xfloor = floorf(x);
      xinteger[kx] = (int) xfloor;
      xfrac[kx] = x-xfloor;
   }

   float yinterp[heightOut];
   int yinteger[heightOut];
   float yfrac[heightOut];
   float dy = (float) (heightIn-1)/(float) (heightOut-1);
   for (int ky=0; ky<heightOut; ky++) {
      float y = dy * (float) ky;
      yinterp[ky] = y;
      float yfloor = floorf(y);
      yinteger[ky] = (int) yfloor;
      yfrac[ky] = y-yfloor;
   }

   memset(bufferOut, 0, sizeof(*bufferOut)*size_t(widthOut*heightOut*numBands));
   for (int xOff = 2; xOff > -2; xOff--) {
      for (int yOff = 2; yOff > -2; yOff--) {
         for (int ky=0; ky<heightOut; ky++) {
            float ycoeff = bicubic(yfrac[ky]-(float) yOff);
            int yfetch = yinteger[ky]+yOff;
            if (yfetch < 0) yfetch = -yfetch;
            if (yfetch >= heightIn) yfetch = heightIn - (yfetch - heightIn) - 1;
            for (int kx=0; kx<widthOut; kx++) {
               float xcoeff = bicubic(xfrac[kx]-(float) xOff);
               int xfetch = xinteger[kx]+xOff;
               if (xfetch < 0) xfetch = -xfetch;
               if (xfetch >= widthIn) xfetch = widthIn - (xfetch - widthIn) - 1;
               assert(xfetch >= 0 && xfetch < widthIn && yfetch >= 0 && yfetch < heightIn);
               for (int f=0; f<numBands; f++) {
                  int fetchIdx = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
                  pvadata_t p = bufferIn[fetchIdx];
                  int outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
                  bufferOut[outputIdx] += xcoeff * ycoeff * p;
               }
            }
         }
      }
   }
   return PV_SUCCESS;
}

int BaseInput::calcLocalBox(int rank, int * dataLeft, int * dataTop, int * imageLeft, int * imageTop, int * width, int * height) {
   Communicator * icComm = parent->icCommunicator();
   PVLayerLoc const * loc = getLayerLoc();
   PVHalo const * halo = &loc->halo;
   int column = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
   int boxInImageLeft = getOffsetX(this->offsetAnchor, this->offsets[0]) + column * getLayerLoc()->nx;
   int boxInImageRight = boxInImageLeft + loc->nx;
   int boxInLayerLeft = halo->lt;
   int boxInLayerRight = boxInLayerLeft + loc->nx;
   int row = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
   int boxInImageTop = getOffsetY(this->offsetAnchor, this->offsets[1]) + row * getLayerLoc()->ny;
   int boxInImageBottom = boxInImageTop + loc->ny;
   int boxInLayerTop = halo->up;
   int boxInLayerBottom = boxInLayerTop + loc->ny;

   if (useImageBCflag) {
      boxInLayerLeft -= halo->lt;
      boxInLayerRight += halo->rt;
      boxInImageLeft -= halo->lt;
      boxInImageRight += halo->rt;

      boxInLayerTop -= halo->up;
      boxInLayerBottom += halo->dn;
      boxInImageTop -= halo->up;
      boxInImageBottom += halo->dn;
   }

   int status = PV_SUCCESS;
   if (boxInImageLeft > imageLoc.nxGlobal || boxInImageRight < 0 ||
       boxInImageTop > imageLoc.nyGlobal || boxInImageBottom < 0 ||
       boxInLayerLeft > loc->nx+halo->lt+halo->rt || boxInLayerRight < 0 ||
       boxInLayerTop > loc->ny+halo->dn+halo->up || boxInLayerBottom < 0) {
      *width = 0;
      *height = 0;
      status = PV_FAILURE;
   }
   else {
      if (boxInImageLeft < 0) {
         int discrepancy = -boxInImageLeft;
         boxInLayerLeft += discrepancy;
         boxInImageLeft += discrepancy;
      }
      if (boxInLayerLeft < 0) {
         int discrepancy = -boxInLayerLeft;
         boxInLayerLeft += discrepancy;
         boxInImageLeft += discrepancy;
      }

      if (boxInImageRight > imageLoc.nxGlobal) {
         int discrepancy = boxInImageRight - imageLoc.nxGlobal;
         boxInLayerRight -= discrepancy;
         boxInImageRight -= discrepancy;
      }
      if (boxInLayerRight > loc->nx+halo->lt+halo->rt) {
         int discrepancy = boxInLayerRight - (loc->nx+halo->lt+halo->rt);
         boxInLayerRight -= discrepancy;
         boxInImageRight -= discrepancy;
      }

      if (boxInImageTop < 0) {
         int discrepancy = -boxInImageTop;
         boxInLayerTop += discrepancy;
         boxInImageTop += discrepancy;
      }
      if (boxInLayerTop < 0) {
         int discrepancy = -boxInLayerTop;
         boxInLayerTop += discrepancy;
         boxInImageTop += discrepancy;
      }

      if (boxInImageBottom > imageLoc.nyGlobal) {
         int discrepancy = boxInImageBottom - imageLoc.nyGlobal;
         boxInLayerBottom -= discrepancy;
         boxInImageBottom -= discrepancy;
      }
      if (boxInLayerBottom > loc->ny+halo->dn+halo->up) {
         int discrepancy = boxInLayerRight - (loc->ny+halo->dn+halo->up);
         boxInLayerBottom -= discrepancy;
         boxInImageBottom -= discrepancy;
      }

      int boxWidth = boxInImageRight-boxInImageLeft;
      int boxHeight = boxInImageBottom-boxInImageTop;
      pvAssert(boxWidth>=0);
      pvAssert(boxHeight>=0);
      pvAssert(boxWidth == boxInLayerRight-boxInLayerLeft);
      pvAssert(boxHeight == boxInLayerBottom-boxInLayerTop);
      // coordinates can be outside of the imageLoc limits, as long as the box has zero area
      pvAssert(boxInImageLeft >= 0 && boxInImageRight <= imageLoc.nxGlobal);
      pvAssert(boxInImageTop >= 0 && boxInImageBottom <= imageLoc.nyGlobal);
      pvAssert(boxInLayerLeft >= 0 && boxInLayerRight <= loc->nx+halo->lt+halo->rt);
      pvAssert(boxInLayerTop >= 0 && boxInLayerBottom <= loc->ny+halo->dn+halo->up);

      *dataLeft = boxInLayerLeft;
      *dataTop = boxInLayerTop;
      *imageLeft = boxInImageLeft;
      *imageTop = boxInImageTop;
      *width = boxWidth;
      *height = boxHeight;
   }
   return status;
}

int BaseInput::copyFromInteriorBuffer(float * buf, int batchIdx, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   const PVHalo * halo = &loc->halo;
   pvdata_t * dataBatch = data + batchIdx * (nx + halo->lt + halo->rt) * (ny + halo->up + halo->dn) * nf;

   if(useImageBCflag){
      for(int n=0; n<getNumExtended(); n++) {
         //int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
         dataBatch[n] = fac*buf[n];
      }
   }else{
      for(int n=0; n<getNumNeurons(); n++) {
         int n_ex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
         dataBatch[n_ex] = fac*buf[n];
      }
   }

   return 0;
}

int BaseInput::copyToInteriorBuffer(unsigned char * buf, int batchIdx, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const PVHalo * halo = &loc->halo;

   pvdata_t * dataBatch = data + batchIdx * (nx + halo->lt + halo->rt) * (ny + halo->up + halo->dn) * nf;
   for(int n=0; n<getNumNeurons(); n++) {
      int n_ex = kIndexExtended(n, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      buf[n] = (unsigned char) (fac * dataBatch[n_ex]);
   }
   return 0;
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
      float* buf = data + b * numExtended;
      if(normalizeLuminanceFlag){
         if (normalizeStdDev){
            double image_sum = 0.0f;
            double image_sum2 = 0.0f;
            for (int k=0; k<numExtended; k++) {
               image_sum += buf[k];
               image_sum2 += buf[k]*buf[k];
            }
            double image_ave = image_sum / numExtended;
            double image_ave2 = image_sum2 / numExtended;
#ifdef PV_USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &image_ave, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
            image_ave /= parent->icCommunicator()->commSize();
            MPI_Allreduce(MPI_IN_PLACE, &image_ave2, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
            image_ave2 /= parent->icCommunicator()->commSize();
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
            MPI_Allreduce(MPI_IN_PLACE, &image_max, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
            MPI_Allreduce(MPI_IN_PLACE, &image_min, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
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
      if( inverseFlag ) {
         for (int k=0; k<numExtended; k++) {
            buf[k] = 1 - buf[k]; // If normalizeLuminanceFlag is true, should the effect of inverseFlag be buf[k] = -buf[k]?
         }
      }
   }
   return PV_SUCCESS;
}

int BaseInput::exchange()
{
   return parent->icCommunicator()->exchange(data, mpi_datatypes, getLayerLoc());
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
      return ((imageLoc.nxGlobal/2)-(layerSizeX/2)) + offsetX;
   }
   //Offset on bottom
   else if(!strcmp(offsetAnchor, "tr") || !strcmp(offsetAnchor, "cr") || !strcmp(offsetAnchor, "br")){
      int layerSizeX = getLayerLoc()->nxGlobal;
      return (imageLoc.nxGlobal - layerSizeX) + offsetX;
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
      return ((imageLoc.nyGlobal/2)-(layerSizeY/2)) + offsetY;
   }
   //Offset on bottom
   else if(!strcmp(offsetAnchor, "bl") || !strcmp(offsetAnchor, "bc") || !strcmp(offsetAnchor, "br")){
      int layerSizeY = getLayerLoc()->nyGlobal;
      return (imageLoc.nyGlobal-layerSizeY) + offsetY;
   }
   assert(0); // All possible cases should be covered above
   return -1; // Eliminates no-return warning
}



//Jitter Methods
//TODO: fix this

/*
 * jitter() is not called by Image directly, but it is called by
 * its derived classes Patterns and Movie, so it's placed in Image.
 * It returns true if the offsets changed so that a new image needs
 * to be loaded/drawn.
 */
bool BaseInput::jitter() {
   // move bias
   double timed = parent->simulationTime();
   if( timed > parent->getStartTime() && timed >= nextBiasChange ){
      calcNewBiases(stepSize);
      constrainBiases();
      nextBiasChange += biasChangeTime;
   }

   // move offset
   bool needNewImage = calcNewOffsets(stepSize);
   constrainOffsets();

   if(writePosition && parent->icCommunicator()->commRank()==0){
      fprintf(fp_pos->fp,"t=%f, bias x=%d, y=%d, offset x=%d y=%d\n",timed,biases[0],biases[1],getOffsetX(this->offsetAnchor, offsets[0]), getOffsetY(this->offsetAnchor, offsets[1]));
   }
   lastUpdateTime = timed;
   return needNewImage;
}

/**
 * Calculate a bias in x or y here.  Input argument is the step size and the size of the interval of possible values
 * Output is the value of the bias.
 * It can perform a random walk of a fixed stepsize or it can perform a random jump up to a maximum length
 * equal to step.
 */
int BaseInput::calcBias(int current_bias, int step, int sizeLength)
{
   assert(jitterFlag);
   double p;
   int dbias = 0;
   if (jitterType == RANDOM_WALK) {
      p = randState->uniformRandom();
      dbias = p < 0.5 ? step : -step;
   } else if (jitterType == RANDOM_JUMP) {
      p = randState->uniformRandom();
      dbias = (int) floor(p*(double) step) + 1;
      p = randState->uniformRandom();
      if (p < 0.5) dbias = -dbias;
   }
   else {
      assert(0); // Only allowable values of jitterType are RANDOM_WALK and RANDOM_JUMP
   }

   int new_bias = current_bias + dbias;
   new_bias = (new_bias < 0) ? -new_bias : new_bias;
   new_bias = (new_bias > sizeLength) ? sizeLength - (new_bias-sizeLength) : new_bias;
   return new_bias;
}

int BaseInput::calcNewBiases(int stepSize) {
   assert(jitterFlag);
   int step_radius = 0; // distance to step
   switch (jitterType) {
   case RANDOM_WALK:
      step_radius = stepSize;
      break;
   case RANDOM_JUMP:
      step_radius = 1 + (int) floor(randState->uniformRandom() * stepSize);
      break;
   default:
      assert(0); // Only allowable values of jitterType are RANDOM_WALK and RANDOM_JUMP
      break;
   }
   double p = randState->uniformRandom() * 2 * PI; // direction to step
   int dx = (int) floor( step_radius * cos(p));
   int dy = (int) floor( step_radius * sin(p));
   assert(dx != 0 || dy != 0);
   biases[0] += dx;
   biases[1] += dy;
   return PV_SUCCESS;
}

/**
 * Return an offset that moves randomly around position bias
 * Perform a
 * random jump of maximum length equal to step.
 * The routine returns the resulting offset.
 * (The recurrenceProb test has been moved to the calling routine jitter() )
 */
int BaseInput::calcBiasedOffset(int bias, int current_offset, int step, int sizeLength)
{
   assert(jitterFlag); // calcBiasedOffset should only be called when jitterFlag is true
   int new_offset;
   double p = randState->uniformRandom();
   int d_offset = (int) floor(p*(double) step) + 1;
   p = randState->uniformRandom();
   if (p<0.5) d_offset = -d_offset;
   new_offset = current_offset + d_offset;
   new_offset = (new_offset < 0) ? -new_offset : new_offset;
   new_offset = (new_offset > sizeLength) ? sizeLength - (new_offset-sizeLength) : new_offset;

   return new_offset;
}

bool BaseInput::calcNewOffsets(int stepSize)
{
   assert(jitterFlag);

   bool needNewImage = false;
   double p = randState->uniformRandom();
   if (timeSinceLastJitter >= jitterRefractoryPeriod) {
      if (p > recurrenceProb) {
         p = randState->uniformRandom();
         if (p > persistenceProb) {
            needNewImage = true;
           int step_radius = 1 + (int) floor(randState->uniformRandom() * stepSize);
           double p = randState->uniformRandom() * 2 * PI; // direction to step
           int dx = (int) round( step_radius * cos(p));
           int dy = (int) round( step_radius * sin(p));
           assert(dx != 0 || dy != 0);
           offsets[0] += dx;
           offsets[1] += dy;
           timeSinceLastJitter = 0;
         }
      }
      else {
            assert(sizeof(*offsets) == sizeof(*biases));
            memcpy(offsets, biases, 2*sizeof(offsets));
            timeSinceLastJitter = 0;
      }
   }
   timeSinceLastJitter++;
   return needNewImage;
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

bool BaseInput::constrainBiases() {
   return constrainPoint(biases, stepSize, imageLoc.nxGlobal - getLayerLoc()->nxGlobal - stepSize, stepSize, imageLoc.nyGlobal - getLayerLoc()->nyGlobal - stepSize, biasConstraintMethod);
}

bool BaseInput::constrainOffsets() {
   int newOffsets[2];
   int oldOffsetX = getOffsetX(this->offsetAnchor, offsets[0]);
   int oldOffsetY = getOffsetY(this->offsetAnchor, offsets[1]);
   newOffsets[0] = oldOffsetX; 
   newOffsets[1] = oldOffsetY; 
   bool status = constrainPoint(newOffsets, 0, imageLoc.nxGlobal - getLayerLoc()->nxGlobal, 0, imageLoc.nyGlobal - getLayerLoc()->nyGlobal, biasConstraintMethod);
   int diffx = newOffsets[0] - oldOffsetX;
   int diffy = newOffsets[1] - oldOffsetY;
   offsets[0] = offsets[0] + diffx;
   offsets[1] = offsets[1] + diffy;
   return status;
}

int BaseInput::requireChannel(int channelNeeded, int * numChannelsResult) {
   if (parent->columnId()==0) {
      pvErrorNoExit().printf("%s cannot be a post-synaptic layer.\n",
            getDescription_c());
   }
   *numChannelsResult = 0;
   return PV_FAILURE;
}

int BaseInput::initRandState() {
   assert(randState==NULL);
   randState = new Random(parent, 1);
   if (randState==NULL) {
      pvError().printf("%s: rank %d process unable to create object of class Random.\n", getDescription_c(), parent->columnId());
   }
   return PV_SUCCESS;
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

#ifdef PV_USE_OPENCL
// no need for threads for now for image
//
int BaseInput::initializeThreadBuffers(const char * kernelName)
{
   return CL_SUCCESS;
}

// no need for threads for now for image
//
int BaseInput::initializeThreadKernels(const char * kernelName)
{
   return CL_SUCCESS;
}
#endif

///**
// * return some useful information about the image
// */
//int BaseInput::tag()
//{
//   return 0;
//}


int BaseInput::checkpointRead(const char * cpDir, double * timeptr){
   PVParams * params = parent->parameters();
   if (parent->columnId()==0) {
      pvWarn().printf("Initializing image from checkpoint NOT from params file location! \n");
   }
   HyPerLayer::checkpointRead(cpDir, timeptr);

   return PV_SUCCESS;
}

int BaseInput::updateState(double time, double dt){
   //Do nothing
   return PV_SUCCESS;
}

/**
 *
 * The data buffer lives in the extended space. Here, we only copy the restricted space
 * to the buffer buf. The size of this buffer is the size of the image patch - borders
 * are not included.
 *
 */
int BaseInput::writeImage(const char * filename, int batchIdx)
{
   int status = 0;
   const PVLayerLoc * loc = getLayerLoc();

   const int n = loc->nx * loc->ny * loc->nf;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   status = copyToInteriorBuffer(buf, batchIdx, 255.0);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf, parent->getVerifyWrites());

   delete[] buf;

   return status;
}


int BaseInput::convertToGrayScale(float ** buffer, int nx, int ny, int numBands, InputColorType colorType)
{
   // even though the numBands argument goes last, the routine assumes that
   // the organization of buf is, bands vary fastest, then x, then y.
   if (numBands < 2) {return PV_SUCCESS;}

   const int sxcolor = numBands;
   const int sycolor = numBands*nx;
   const int sb = 1;

   const int sxgray = 1;
   const int sygray = nx;

   float * graybuf = new float[nx*ny];
   float * colorbuf = *buffer;

   float bandweight[numBands];
   calcBandWeights(numBands, bandweight, colorType);

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = colorbuf[i*sxcolor + j*sycolor + b*sb];
            val += d*bandweight[b];
         }
         graybuf[i*sxgray + j*sygray] = val;
      }
   }
   delete[] *buffer;
   *buffer = graybuf;
   return PV_SUCCESS;
}

int BaseInput::convertGrayScaleToMultiBand(float ** buffer, int nx, int ny, int numBands)
{
   const int sxcolor = numBands;
   const int sycolor = numBands*nx;
   const int sb = 1;

   const int sxgray = 1;
   const int sygray = nx;

   float * multiBandsBuf = new float[nx*ny*numBands];
   float * graybuf = *buffer;

   for (int j = 0; j < ny; j++)
   {
      for (int i = 0; i < nx; i++)
      {
         for (int b = 0; b < numBands; b++)
         {
            multiBandsBuf[i*sxcolor + j*sycolor + b*sb] = graybuf[i*sxgray + j*sygray];
         }

      }
   }
   delete[] *buffer;
   *buffer = multiBandsBuf;
   return PV_SUCCESS;
}

int BaseInput::calcBandWeights(int numBands, float * bandweight, InputColorType colorType) {
   const float grayalphaweights[2] = {1.0, 0.0};
   const float rgbaweights[4] = {0.30f, 0.59f, 0.11f, 0.0f}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
   switch (colorType) {
   case COLORTYPE_UNRECOGNIZED:
      equalBandWeights(numBands, bandweight);
      break;
   case COLORTYPE_GRAYSCALE:
      if (numBands==1 || numBands==2) {
         memcpy(bandweight, grayalphaweights, numBands*sizeof(*bandweight));
      }
      else {
         pvAssert(0);
      }
      break;
   case COLORTYPE_RGB:
      if (numBands==3 || numBands==4) {
         memcpy(bandweight, rgbaweights, numBands*sizeof(*bandweight));
      }
      else {
         pvAssert(0);
      }
      break;
   default:
      pvAssert(0);
   }
   return PV_SUCCESS;
}

} // namespace PV





