/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"

#ifdef PV_USE_MPI
#include <mpi.h>
#endif
#include <assert.h>
#include <string.h>

namespace PV {

Image::Image() {
   initialize_base();
}

Image::Image(const char * name, HyPerCol * hc, const char * filename) {
   initialize_base();
   initialize(name, hc, filename);
}

Image::~Image() {
   free(filename);
   filename = NULL;
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   delete randState; randState = NULL;

   if(writePosition){
      if (getParent()->icCommunicator()->commRank()==0 && fp_pos != NULL && fp_pos->isfile) {
         PV_fclose(fp_pos);
      }
   }
}

int Image::initialize_base() {
   mpi_datatypes = NULL;
   data = NULL;
   filename = NULL;
   imageData = NULL;
   useImageBCflag = false;
   autoResizeFlag = false;
   writeImages = false;
   writeImagesExtension = NULL;
   inverseFlag = false;
   normalizeLuminanceFlag = false;
   offsets[0] = 0;
   offsets[1] = 0;
   jitterFlag = false;
   jitterType = RANDOM_WALK;
   timeSinceLastJitter = 0;
   jitterRefractoryPeriod = 0;
   stepSize = 0;
   persistenceProb = 0.0;
   recurrenceProb = 1.0;
   biasChangeTime = LONG_MAX;
   writePosition = 0;
   fp_pos = NULL;
   biases[0]   = getOffsetX();
   biases[1]   = getOffsetY();
   frameNumber = 0;
   randState = NULL;
   return PV_SUCCESS;
}

int Image::initialize(const char * name, HyPerCol * hc, const char * filename) {
   if (filename != NULL) {
      this->filename = strdup(filename);
      assert( this->filename != NULL);
   }
   else {
      this->filename = NULL;
   }

   HyPerLayer::initialize(name, hc, 0);
   // Much of the functionality that was previously here has been moved to either read-methods, communicateInitInfo, or allocateDataStructures

   int status = PV_SUCCESS;

   this->lastUpdateTime = 0.0;

   PVParams * params = parent->parameters();

   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      assert(!params->presentAndNotBeenRead(name, "offsetX"));
      assert(!params->presentAndNotBeenRead(name, "offsetY"));
      biases[0] = getOffsetX();
      biases[1] = getOffsetY();
   }

   // TODO - Add option to use image size to determine layer size and column size

   return status;
}

int Image::setParams(PVParams * params) {
   int status = HyPerLayer::setParams(params);

   readOffsets(params);
   readWriteImagesFlag(params);

   readUseImageBCflag(params);
   readAutoResizeFlag(params);
   readInverseFlag(params);
   readNormalizeLuminanceFlag(params);

   readFrameNumber(params);

   // Although Image itself does not use jitter, both Movie and Patterns do, so jitterFlag is read in Image.
   readJitterFlag(params);
   readJitterType(params);
   readJitterRefractoryPeriod(params);
   readStepSize(params);
   readPersistenceProb(params);
   readRecurrenceProb(params);
   readBiasChangeTime(params);
   readBiasConstraintMethod(params);
   readOffsetConstraintMethod(params);
   readWritePosition(params);

   return status;
}

int Image::readOffsets(PVParams * params) {

   offsets[0]      = (int) params->value(name,"offsetX", offsets[0]);
   offsets[1]      = (int) params->value(name,"offsetY", offsets[1]);

   return PV_SUCCESS;
}

void Image::readWriteImagesFlag(PVParams * params) {
   this->writeImages = params->value(name, "writeImages", writeImages) != 0;
}

void Image::readWriteImagesExtension(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "writeImages"));
   if (this->writeImages) {
      if (params->stringPresent(name, "writeImagesExtension")) {
         writeImagesExtension = strdup(params->stringValue(name, "writeImagesExtension", false));
      }
      else {
         writeImagesExtension = strdup("tif");
         if (parent->columnId()==0) {
            fprintf(stderr, "Using default value \"tif\" for parameter \"writeImagesExtension\" in group %s\n", name);
         }
      }
   }
}

void Image::readUseImageBCflag(PVParams * params) {
   this->useImageBCflag = (bool) params->value(name, "useImageBCflag", useImageBCflag);
}

void Image::readAutoResizeFlag(PVParams * params) {
   this->autoResizeFlag = (bool) params->value(name, "autoResizeFlag", autoResizeFlag);
}

void Image::readInverseFlag(PVParams * params) {
   this->inverseFlag = (bool) params->value(name, "inverseFlag", inverseFlag);
}

void Image::readNormalizeLuminanceFlag(PVParams * params) {
   this->normalizeLuminanceFlag = (bool) params->value(name, "normalizeLuminanceFlag", normalizeLuminanceFlag);
}

void Image::readFrameNumber(PVParams * params) {
   if (filename!=NULL && getFileType(filename)==PVP_FILE_TYPE) {
      this->frameNumber = params->value(name, "frameNumber", frameNumber);
   }
}

void Image::readJitterFlag(PVParams * params) {
   jitterFlag = params->value(name, "jitterFlag", 0) != 0;
}

void Image::readJitterType(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      jitterType = params->value(name, "jitterType", jitterType);
   }
}

void Image::readJitterRefractoryPeriod(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      jitterRefractoryPeriod = params->value(name, "jitterRefractoryPeriod", jitterRefractoryPeriod);
   }
}

void Image::readStepSize(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      stepSize = params->value(name, "stepSize", stepSize);
   }
}

void Image::readPersistenceProb(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      persistenceProb = params->value(name, "persistenceProb", persistenceProb);
   }
}

void Image::readRecurrenceProb(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      recurrenceProb = params->value(name, "recurrenceProb", recurrenceProb);
   }
}


void Image::readBiasChangeTime(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      double biasChangeTimeParam = params->value(name, "biasChangeTime", biasChangeTime);
      if (biasChangeTimeParam==FLT_MAX || biasChangeTimeParam < 0) {
         biasChangeTime = LONG_MAX;
      }
      else {
         biasChangeTime = (long) biasChangeTimeParam;
      }
   }
}

void Image::readBiasConstraintMethod(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      biasConstraintMethod = params->value(name, "biasConstraintMethod", biasConstraintMethod);
      if (biasConstraintMethod <0 || biasConstraintMethod >3) {
         fprintf(stderr, "Image layer \"%s\": biasConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getName());
         exit(EXIT_FAILURE);
      }
   }
}

void Image::readOffsetConstraintMethod(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      offsetConstraintMethod = params->value(name, "offsetConstraintMethod",0);
      if (offsetConstraintMethod <0 || offsetConstraintMethod >3) {
         fprintf(stderr, "Image layer \"%s\": offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getName());
         exit(EXIT_FAILURE);
      }
   }
}

void Image::readWritePosition(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      writePosition     = (int) params->value(name,"writePosition", writePosition);
   }
}

int Image::communicateInitInfo() {
   return HyPerLayer::communicateInitInfo();
}

int Image::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   randState = new Random(parent, 1);
   if (randState==NULL) {
      fprintf(stderr, "%s \"%s\" error in rank %d process: unable to create object of class Random.\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId());
      exit(EXIT_FAILURE);
   }

   // Moved to allocateV()
   // free(clayer->V);
   // clayer->V = NULL;

   data = clayer->activity->data;

   if(filename != NULL) {
      GDALColorInterp * colorbandtypes = NULL;
      status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if( getLayerLoc()->nf != imageLoc.nf && getLayerLoc()->nf != 1) {
         fprintf(stderr, "Image %s: file %s has %d features but the layer has %d features.  Exiting.\n",
               name, filename, imageLoc.nf, getLayerLoc()->nf);
         exit(PV_FAILURE);
      }
      status = readImage(filename, getOffsetX(), getOffsetY(), colorbandtypes);
      assert(status == PV_SUCCESS);
      free(colorbandtypes); colorbandtypes = NULL;
   }
   else {
      this->imageLoc = * getLayerLoc();
   }

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
            fprintf(stderr, "Path for jitter positions \"%s/%s_jitter.txt is too long.\n", parent->getOutputPath(), getName());
            abort();
         }
         printf("Image layer \"%s\" will write jitter positions to %s\n",getName(), file_name);
         fp_pos = PV_fopen(file_name,"w");
         if(fp_pos == NULL) {
            fprintf(stderr, "Image \"%s\" unable to open file \"%s\" for writing jitter positions.\n", getName(), file_name);
            abort();
         }
         fprintf(fp_pos->fp,"Layer \"%s\", t=%f, bias x=%d y=%d, offset x=%d y=%d\n",getName(),parent->simulationTime(),biases[0],biases[1],
               getOffsetX(),getOffsetY());
      }
   }

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

   // exchange border information
   exchange();

   return status;
}

int Image::allocateV() {
   clayer->V = NULL;
   return PV_SUCCESS;
}

int Image::initializeState() {
   int status = PV_SUCCESS;

   PVParams * params = parent->parameters();
   assert(!params->presentAndNotBeenRead(name, "restart"));
   if( restartFlag ) {
      double timef;
      status = readState(&timef);
   }
   return status;
}

#ifdef PV_USE_OPENCL
// no need for threads for now for image
//
int Image::initializeThreadBuffers(const char * kernelName)
{
   return CL_SUCCESS;
}

// no need for threads for now for image
//
int Image::initializeThreadKernels(const char * kernelName)
{
   return CL_SUCCESS;
}
#endif

/**
 * return some useful information about the image
 */
int Image::tag()
{
   return 0;
}

int Image::recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int neighbor)
{
   // this should never be called as an image shouldn't have an incoming connection
   recvsyn_timer->start();
   recvsyn_timer->stop();
   return 0;
}

/**
 * update the image buffers
 */
int Image::updateState(double time, double dt)
{
   // make sure image is copied to activity buffer
   //
   //update_timer->start();
   //update_timer->stop();
   return 0;
}

int Image::outputState(double time, bool last)
{
   //io_timer->start();
   // this could probably use Marion's update time interval
   // for some classes
   //
   //io_timer->stop();

   return 0;
}

int Image::checkpointRead(const char * cpDir, double * timef){

   PVParams * params = parent->parameters();
   this->useParamsImage      = (int) params->value(name,"useParamsImage", 0);
   if (this->useParamsImage) {
      fprintf(stderr,"Initializing image from params file location ! \n");
      * timef = parent->simulationTime(); // fakes the pvp time stamp
   }
   else {
      fprintf(stderr,"Initializing image from checkpoint NOT from params file location! \n");
      HyPerLayer::checkpointRead(cpDir, timef);
   }

   return PV_SUCCESS;
}


//! CLEAR IMAGE
/*!
 * this is Image specific.
 */
int Image::clearImage()
{
   // default is to do nothing for now
   // it could, for example, set the data buffer to zero.

   return 0;
}

int Image::readImage(const char * filename)
{
   return readImage(filename, 0, 0, NULL);
}

int Image::readImage(const char * filename, int offsetX, int offsetY, GDALColorInterp * colorbandtypes)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   if(useImageBCflag){ //Expand dimensions to the extended space
      loc->nx = loc->nx + 2*loc->nb;
      loc->ny = loc->ny + 2*loc->nb;
   }

   int n = loc->nx * loc->ny * imageLoc.nf;

   // Use number of bands in file instead of in params, to allow for grayscale conversion
   float * buf = new float[n];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(filename, offsetX, offsetY, parent->icCommunicator(), loc, buf, frameNumber, this->autoResizeFlag);
   assert(status == PV_SUCCESS);
   if( loc->nf == 1 && imageLoc.nf > 1 ) {
      float * graybuf = convertToGrayScale(buf,loc->nx,loc->ny,imageLoc.nf, colorbandtypes);
      delete[] buf;
      buf = graybuf;
      //Redefine n for grayscale images
      n = loc->nx * loc->ny;
   }
   else if (loc->nf > 1 && imageLoc.nf == 1)
   {
       buf = copyGrayScaletoMultiBands(buf,loc->nx,loc->ny,loc->nf,colorbandtypes);
       n = loc->nx * loc->ny * loc->nf;
       
   }
   
   // now buf is loc->nf by loc->nx by loc->ny

   // if normalizeLuminanceFlag == true then force average luminance to be 0.5
   if(normalizeLuminanceFlag){
      double image_sum = 0.0f;
      double image_sum2 = 0.0f;
      float image_max = -FLT_MAX;
      float image_min = FLT_MAX;
      for (int k=0; k<n; k++) {
         image_sum += buf[k];
         image_sum2 += buf[k]*buf[k];
         image_max = buf[k] > image_max ? buf[k] : image_max;
         image_min = buf[k] < image_min ? buf[k] : image_min;
      }
      double image_ave = image_sum / n;
      double image_ave2 = image_sum2 / n;
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &image_ave, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      image_ave /= parent->icCommunicator()->commSize();
      MPI_Allreduce(MPI_IN_PLACE, &image_ave2, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      image_ave2 /= parent->icCommunicator()->commSize();
      MPI_Allreduce(MPI_IN_PLACE, &image_max, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &image_min, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
#endif
      //TODO!!! obviously should be passed as Image param or change normalizeLuminanceFlag to an int to specify different behaviors
      bool normalize_standard_dev = true;
      if (normalize_standard_dev){
    	  // set mean to zero
    	  for (int k=0; k<n; k++) {
			  buf[k] -= image_ave;
		  }
    	  // set std dev to 1
    	  double image_std = sqrt(image_ave2 - image_ave*image_ave);
    	  for (int k=0; k<n; k++) {
			  buf[k] /= image_std;
		  }
      } // normalize_standard_dev
      else{
    	  if (image_max > image_min){
    		  float image_stretch = 1.0f / (image_max - image_min);
    		  for (int k=0; k<n; k++) {
    			  buf[k] -= image_min;
    			  buf[k] *= image_stretch;
    		  }
    	  }
    	  else{ // image_max == image_min
    		  float image_shift = 0.5f - image_ave;
    		  for (int k=0; k<n; k++) {
    			  buf[k] += image_shift;
    		  }
    	  }
      }  // normalize_standard_dev
   } // normalizeLuminanceFlag

   if( inverseFlag ) {
      for (int k=0; k<n; k++) {
         buf[k] = 1 - buf[k];
      }
   }

   if( status == PV_SUCCESS ) copyFromInteriorBuffer(buf, 1.0f);

   delete[] buf;

   if(useImageBCflag){ //Restore non-extended dimensions
      loc->nx = loc->nx - 2*loc->nb;
      loc->ny = loc->ny - 2*loc->nb;
   }

   return status;
}

/**
 *
 * The data buffer lives in the extended space. Here, we only copy the restricted space
 * to the buffer buf. The size of this buffer is the size of the image patch - borders
 * are not included.
 *
 */
int Image::write(const char * filename)
{
   int status = 0;
   const PVLayerLoc * loc = getLayerLoc();

   const int n = loc->nx * loc->ny * loc->nf;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   status = copyToInteriorBuffer(buf, 255.0);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf);

   delete buf;

   return status;
}

int Image::exchange()
{
   return parent->icCommunicator()->exchange(data, mpi_datatypes, getLayerLoc());
}


int Image::copyToInteriorBuffer(unsigned char * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nBorder = loc->nb;

   for(int n=0; n<getNumNeurons(); n++) {
      int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
      buf[n] = (unsigned char) (fac * data[n_ex]);
   }
   return 0;
}

int Image::copyFromInteriorBuffer(float * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   const int nBorder = loc->nb;

   if(useImageBCflag){
      for(int n=0; n<getNumExtended(); n++) {
         //int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
         data[n] = fac*buf[n];
      }
   }else{
      for(int n=0; n<getNumNeurons(); n++) {
         int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
         data[n_ex] = fac*buf[n];
      }
   }

   return 0;
}

float * Image::convertToGrayScale(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes)
{
   // even though the numBands argument goes last, the routine assumes that
   // the organization of buf is, bands vary fastest, then x, then y.
   if (numBands < 2) return buf;


   const int sxcolor = numBands;
   const int sycolor = numBands*nx;
   const int sb = 1;

   const int sxgray = 1;
   const int sygray = nx;

   float * graybuf = new float[nx*ny];

   float * bandweight = (float *) malloc(numBands*sizeof(float));
   calcBandWeights(numBands, bandweight, colorbandtypes);

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = buf[i*sxcolor + j*sycolor + b*sb];
            val += d*bandweight[b];
         }
         graybuf[i*sxgray + j*sygray] = val;
      }
   }
   free(bandweight);
   return graybuf;
}
    float* Image::copyGrayScaletoMultiBands(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes)
    {
        const int sxcolor = numBands;
        const int sycolor = numBands*nx;
        const int sb = 1;

        const int sxgray = 1;
        const int sygray = nx;

        float * multiBandsBuf = new float[nx*ny*numBands];
        
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int b = 0; b < numBands; b++)
                {
                    multiBandsBuf[i*sxcolor + j*sycolor + b*sb] = buf[i*sxgray + j*sygray];
                }

            }

        }
        delete buf;
        return multiBandsBuf;
                
    }
    
int Image::calcBandWeights(int numBands, float * bandweight, GDALColorInterp * colorbandtypes) {
   int colortype = 0; // 1=grayscale(with or without alpha), return value 2=RGB(with or without alpha), 0=unrecognized
   const GDALColorInterp grayalpha[2] = {GCI_GrayIndex, GCI_AlphaBand};
   const GDALColorInterp rgba[4] = {GCI_RedBand, GCI_GreenBand, GCI_BlueBand, GCI_AlphaBand};
   const float grayalphaweights[2] = {1.0, 0.0};
   const float rgbaweights[4] = {0.30, 0.59, 0.11, 0.0}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
   switch( numBands ) {
   case 1:
      bandweight[0] = 1.0;
      colortype = 1;
      break;
   case 2:
      if ( !memcmp(colorbandtypes, grayalpha, 2*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, grayalphaweights, 2*sizeof(float));
         colortype = 1;
      }
      break;
   case 3:
      if ( !memcmp(colorbandtypes, rgba, 3*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 3*sizeof(float));
         colortype = 2;
      }
      break;
   case 4:
      if ( !memcmp(colorbandtypes, rgba, 4*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 4*sizeof(float));
         colortype = 2;
      }
      break;
   default:
      break;
   }
   if (colortype==0) {
      equalBandWeights(numBands, bandweight);
   }
   return colortype;
}

void Image::equalBandWeights(int numBands, float * bandweight) {
   float w = 1.0/(float) numBands;
   for( int b=0; b<numBands; b++ ) bandweight[b] = w;
}

/*
 * jitter() is not called by Image directly, but it is called by
 * its derived classes Patterns and Movie, so it's placed in Image.
 * It returns true if the offsets changed so that a new image needs
 * to be loaded/drawn.
 */
bool Image::jitter() {
   // move bias
   double timed = parent->simulationTime();
   if( timed > 0 && !( ((long)timed) % biasChangeTime ) ){  // Needs to be changed: dt is not always 1.
      calcNewBiases(stepSize);
      constrainBiases();
   }

   // move offset
   bool needNewImage = calcNewOffsets(stepSize);
   constrainOffsets();

   if(writePosition && parent->icCommunicator()->commRank()==0){
      fprintf(fp_pos->fp,"t=%f, bias x=%d, y=%d, offset x=%d y=%d\n",timed,biases[0],biases[1],offsets[0],offsets[1]);
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
int Image::calcBias(int current_bias, int step, int sizeLength)
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

int Image::calcNewBiases(int stepSize) {
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
int Image::calcBiasedOffset(int bias, int current_offset, int step, int sizeLength)
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

bool Image::calcNewOffsets(int stepSize)
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

bool Image::constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method) {
   bool moved_x = point[0] < min_x || point[0] > max_x;
   bool moved_y = point[1] < min_y || point[1] > max_y;
   if (moved_x) {
      if (min_x >= max_x) {
         fprintf(stderr, "Image::constrainPoint error.  min_x=%d and max_x= %d\n", min_x, max_x);
         abort();
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
         assert(0);
         break;
      }
      assert(new_x >= min_x && new_x <= max_x);
      point[0] = new_x;
   }
   if (moved_y) {
      if (min_y >= max_y) {
         fprintf(stderr, "Image::constrainPoint error.  min_y=%d and max_y=%d\n", min_y, max_y);
         abort();
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
         if (new_y>=size_y) new_y = 2*size_y+1-new_y;
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

bool Image::constrainBiases() {
   return constrainPoint(biases, 0, imageLoc.nxGlobal - getLayerLoc()->nxGlobal, 0, imageLoc.nyGlobal - getLayerLoc()->nyGlobal - stepSize, biasConstraintMethod);
}

bool Image::constrainOffsets() {
   return constrainPoint(offsets, 0, imageLoc.nxGlobal - getLayerLoc()->nxGlobal, 0, imageLoc.nyGlobal - getLayerLoc()->nyGlobal - stepSize, biasConstraintMethod);
}

} // namespace PV
