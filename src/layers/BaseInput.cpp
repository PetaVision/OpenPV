/*
 * BaseInput.cpp
 */

#include "BaseInput.hpp"

namespace PV {

BaseInput::BaseInput() {
   initialize_base();
}

//Image::Image(const char * name, HyPerCol * hc) {
//   initialize_base();
//   initialize(name, hc);
//}

BaseInput::~BaseInput() {
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   delete randState; randState = NULL;

   if(writePosition){
      if (getParent()->icCommunicator()->commRank()==0 && fp_pos != NULL && fp_pos->isfile) {
         PV_fclose(fp_pos);
      }
   }
   if(offsetAnchor){
      free(offsetAnchor);
   }
   free(writeImagesExtension);
   if(inputPath){
      free(inputPath);
   }
}

int BaseInput::initialize_base() {
   numChannels = 0;
   mpi_datatypes = NULL;
   data = NULL;
   //filename = NULL;
   imageData = NULL;
   useImageBCflag = false;
   //autoResizeFlag = false;
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
   //frameNumber = 0;
   randState = NULL;
   //posstream = NULL;
   //pvpFileTime = 0;
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
            fprintf(stderr, "%s \"%s\" error: offsetAnchor must be a two-letter string.  The first character must be \"t\", \"c\", or \"b\" (for top, center or bottom); and the second character must be \"l\", \"c\", or \"r\" (for left, center or right).\n", getKeyword(), getName());
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
         fprintf(stderr, "%s \"%s\": biasConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n",
               getKeyword(), getName());
         exit(EXIT_FAILURE);
      }
   }
}

void BaseInput::ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "jitterFlag"));
   if (jitterFlag) {
      parent->ioParamValue(ioFlag, name, "offsetConstraintMethod", &offsetConstraintMethod, 0/*default*/);
      if (ioFlag == PARAMS_IO_READ && (offsetConstraintMethod <0 || offsetConstraintMethod >3) ) {
         fprintf(stderr, "Image layer \"%s\": offsetConstraintMethod allowed values are 0 (ignore), 1 (mirror BC), 2 (threshold), 3 (circular BC)\n", getName());
         exit(EXIT_FAILURE);
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

   ////Image copies 
   //assert(imageFilename);
   
   //status = readImage(imageFilename, this->offsets[0], this->offsets[1], this->offsetAnchor);
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
            fprintf(stderr, "Path for jitter positions \"%s/%s_jitter.txt is too long.\n", parent->getOutputPath(), getName());
            abort();
         }
         printf("Image layer \"%s\" will write jitter positions to %s\n",getName(), file_name);
         fp_pos = PV_fopen(file_name,"w",parent->getVerifyWrites());
         if(fp_pos == NULL) {
            fprintf(stderr, "Image \"%s\" unable to open file \"%s\" for writing jitter positions.\n", getName(), file_name);
            abort();
         }
         fprintf(fp_pos->fp,"Layer \"%s\", t=%f, bias x=%d y=%d, offset x=%d y=%d\n",getName(),parent->simulationTime(),biases[0],biases[1],
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
int BaseInput::getFrame(double timef, double dt){
   int status = retrieveData(timef, dt);
   assert(status == PV_SUCCESS);
   status = postProcess(timef, dt); //Post processing on activity buffer
   assert(status == PV_SUCCESS);
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

//Post processing done out of this function

//TODO
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
         fprintf(stderr, "Image::constrainPoint error.  min_x=%d is greater than max_x= %d\n", min_x, max_x);
         exit(EXIT_FAILURE);
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
         std::cout << "Method type " << method << " not understood\n";
         assert(0);
         break;
      }
      assert(new_x >= min_x && new_x <= max_x);
      point[0] = new_x;
   }
   if (moved_y) {
      if (min_y > max_y) {
         fprintf(stderr, "Image::constrainPoint error.  min_y=%d is greater than max_y=%d\n", min_y, max_y);
         exit(EXIT_FAILURE);
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
      fprintf(stderr, "%s \"%s\" cannot be a post-synaptic layer.\n",
            getKeyword(), name);
   }
   *numChannelsResult = 0;
   return PV_FAILURE;
}

int BaseInput::initRandState() {
   assert(randState==NULL);
   randState = new Random(parent, 1);
   if (randState==NULL) {
      fprintf(stderr, "%s \"%s\" error in rank %d process: unable to create object of class Random.\n", getKeyword(), name, parent->columnId());
      exit(EXIT_FAILURE);
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
      fprintf(stderr,"Initializing image from checkpoint NOT from params file location! \n");
   }
   HyPerLayer::checkpointRead(cpDir, timeptr);

   return PV_SUCCESS;
}

int BaseInput::updateState(double time, double dt){
   //Do nothing
   return PV_SUCCESS;
}

////! CLEAR IMAGE
///*!
// * this is Image specific.
// */
//int BaseInput::clearImage()
//{
//   // default is to do nothing for now
//   // it could, for example, set the data buffer to zero.
//
//   return 0;
//}

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

}





