/*
 * Patterns.cpp
 *
 *  Created on: April 21, 2010
 *      Author: Marian Anghel and Craig Rasmussen
 */

#include "Patterns.hpp"
#include "../include/pv_common.h"  // for PI

#define PATTERNS_MAXVAL  1.0f
#define PATTERNS_MINVAL  0.0f

namespace PV {

// CER-new

Patterns::Patterns() {
   initialize_base();
}

Patterns::Patterns(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

Patterns::~Patterns()
{
   free(typeString);
   free(orientationString);
   free(patternsOutputPath);
   vDrops.clear();

   if( patternsFile != NULL ) {
      PV_fclose(patternsFile);
      patternsFile = NULL;
   }
}

int Patterns::initialize_base() {
   typeString = NULL;
   orientationString = NULL;
   writePosition = 0;
   patternsOutputPath = NULL;
   patternsFile = NULL;
   framenumber = 0;
   maxVal = 0.0;
   initPatternCntr = 0;
   vDrops.clear();
   orientation = vertical;

   return PV_SUCCESS;
}

int Patterns::initialize(const char * name, HyPerCol * hc) {
   BaseInput::initialize(name, hc);
   assert(getLayerLoc()->nf == 1);
   // this->type is set in Patterns::ioParams, called by HyPerLayer::initialize
   PVParams * params = hc->parameters();
   const PVLayerLoc * loc = getLayerLoc();

   if (type==BARS) {
      position = 0;
   }
   if (type==IMPULSE) {
      initPatternCntr = 0;
   }
   if (type==DROP) {
     //Assign first drop
      //radius.push_back(0);
      //Assign next drop
      if(dropPeriod == -1){
         nextDropFrame = ceil(startFrame);
      }
      else{
         nextDropFrame = dropPeriod;
      }

   }

   if(writePosition){
      char file_name[PV_PATH_MAX];

      //Return value of snprintf commented out because it was generating an
      //unused-variable compiler warning.
      //
      snprintf(file_name, PV_PATH_MAX-1, "%s/patterns-pos.txt", patternsOutputPath);
      //int nchars = snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", patternsOutputPath);
      if (parent->columnId()==0) {
         printf("write position to %s\n",file_name);
         patternsFile = PV_fopen(file_name,"a",parent->getVerifyWrites());
         if(patternsFile == NULL) {
            fprintf(stderr, "Patterns layer \"%s\" unable to open \"%s\" for writing: error %s\n", name, file_name, strerror(errno));
            abort();
         }
      }
      else {
         patternsFile = NULL; // Only root process should write to patternsFile
      }
   }

   // displayPeriod = 0 means nextDisplayTime will always >= starting time and therefore the pattern will update every timestep
   //nextDisplayTime = hc->simulationTime() + displayPeriod;

   return PV_SUCCESS;
}

//void Patterns::ioParam_imagePath(enum ParamsIOFlag ioFlag) {
//   if (ioFlag == PARAMS_IO_READ) {
//      filename = NULL;
//      parent->parameters()->handleUnnecessaryStringParameter(name, "imageList", NULL);
//   }
//}

int Patterns::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);
   ioParam_patternType(ioFlag);
   ioParam_orientation(ioFlag);
   ioParam_pMove(ioFlag);
   ioParam_pSwitch(ioFlag);
   ioParam_movementType(ioFlag);
   ioParam_movementSpeed(ioFlag);
   ioParam_writePosition(ioFlag);
   ioParam_maxValue(ioFlag);
   ioParam_width(ioFlag);
   ioParam_height(ioFlag);
   ioParam_wavelengthVert(ioFlag);
   ioParam_wavelengthHoriz(ioFlag);
   ioParam_rotation(ioFlag);
   ioParam_dropSpeed(ioFlag);
   ioParam_dropSpeedRandomMax(ioFlag);
   ioParam_dropSpeedRandomMin(ioFlag);
   ioParam_dropPeriod(ioFlag);
   ioParam_dropPeriodRandomMax(ioFlag);
   ioParam_dropPeriodRandomMin(ioFlag);
   ioParam_dropPosition(ioFlag);
   ioParam_dropPositionRandomMax(ioFlag);
   ioParam_dropPositionRandomMin(ioFlag);
   ioParam_halfNeutral(ioFlag);
   ioParam_minValue(ioFlag);
   ioParam_inOut(ioFlag);
   ioParam_startFrame(ioFlag);
   ioParam_endFrame(ioFlag);
   ioParam_patternsOutputPath(ioFlag);
   ioParam_displayPeriod(ioFlag);
   return status;
}

int Patterns::stringMatch(const char ** allowed_values, const char * stopstring, const char * string_to_match) {
   int match = -1;
   for( int i=0; strcmp(allowed_values[i], stopstring); i++ ) {
      const char * current_string = allowed_values[i];
      if( !strcmp(string_to_match, current_string) ) {
         match = i;
         break;
      }
   }
   return match;
}

void Patterns::ioParam_patternType(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "patternType", &typeString);
   if (ioFlag==PARAMS_IO_READ) {
      const char * allowed_pattern_types[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
            "BARS",
            "RECTANGLES",
            "SINEWAVE",
            "COSWAVE",
            "IMPULSE",
            "SINEV",
            "COSV",
            "DROP",
            "_End_allowedPatternTypes"  // Keep this string; it allows the string matching loop to know when to stop.
      };
      int match = stringMatch(allowed_pattern_types, "_End_allowedPatternTypes", typeString);
      if( match < 0 ) {
         if (parent->columnId()==0) {
            fprintf(stderr, "Group \"%s\": Pattern type \"%s\" not recognized.\n", name, typeString);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void Patterns::ioParam_orientation(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (!(type==BARS || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV)) {
      return;
   }
   parent->ioParamString(ioFlag, name, "orientation", &orientationString, "VERTICAL");
   if (ioFlag != PARAMS_IO_READ) {
      return;
   }
   const char * allowedOrientationModes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "HORIZONTAL",
         "VERTICAL",
         "MIXED",
         "_End_allowedOrientationTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   int match = stringMatch(allowedOrientationModes, "_End_allowedOrientationTypes", orientationString);
   if (match < 0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Group \"%s\": Orientation mode \"%s\" not recognized.\n", name, orientationString);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   setOrientation((OrientationMode) match);
}

int Patterns::setOrientation(OrientationMode ormode) {
   orientation = ormode;
   switch(orientation) {
   case vertical:
      positionBound = wavelengthVert;
      break;
   case horizontal:
      positionBound = wavelengthHoriz;
      break;
   case mixed:
   default:
      assert(0);
      break;
   }
   return PV_SUCCESS;
}

void Patterns::ioParam_pMove(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "pMove", &pMove, 0.0f);
   }
}

void Patterns::ioParam_pSwitch(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "pSwitch", &pSwitch, 0.0f);
   }
}

void Patterns::ioParam_movementType(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES) {
      parent->ioParamString(ioFlag, name, "movementType", &movementTypeString, "RANDOMWALK"/*default value*/);
      if (ioFlag!=PARAMS_IO_READ) {
         return;
      }
      const char * allowedMovementTypes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
            "RANDOMWALK",
            "MOVEFORWARD",
            "MOVEBACKWARD",
            "RANDOMJUMP",
            "_End_allowedMovementTypes"  // Keep this string; it allows the string matching loop to know when to stop.
      };
      int match = stringMatch(allowedMovementTypes, "_End_allowedMovementTypes", movementTypeString);
      if (match < 0) {
         if (parent->columnId()==0) {
            fprintf(stderr, "Group \"%s\": movementType \"%s\" not recognized.\n", name, movementTypeString);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      movementType = (MovementType) match;
   }
}

void Patterns::ioParam_movementSpeed(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES) {
      parent->ioParamValue(ioFlag, name, "movementSpeed", &movementSpeed, 1.0f);
   }
}

void Patterns::ioParam_writePosition(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "writePosition", &writePosition, 1);
   }
   else {
      writePosition = false;
   }
}

void Patterns::ioParam_maxValue(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV || type==DROP) {
      parent->ioParamValue(ioFlag, name, "maxValue", &maxVal, PATTERNS_MAXVAL);
   }
}

void Patterns::ioParam_width(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "width", &maxWidth, getLayerLoc()->nx);
   }
}

void Patterns::ioParam_height(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==RECTANGLES || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "height", &maxHeight, getLayerLoc()->ny);
   }
}

void Patterns::ioParam_wavelengthVert(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "width"));
      parent->ioParamValue(ioFlag, name, "wavelengthVert", &wavelengthVert, 2*maxWidth);
   }
}

void Patterns::ioParam_wavelengthHoriz(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==BARS || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "height"));
      parent->ioParamValue(ioFlag, name, "wavelengthHoriz", &wavelengthHoriz, 2*maxHeight);
   }
}

void Patterns::ioParam_rotation(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      parent->ioParamValue(ioFlag, name, "rotation", &rotation, 0.0f);
   }
}

void Patterns::ioParam_dropSpeed(enum ParamsIOFlag ioFlag) {
   //(pixels/dt) Radius expands dropSpeed pixles per timestep
   // -1 for random speed: see dropSpeedRandomMax and dropSpeedRandomMin
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
       parent->ioParamValue(ioFlag, name, "dropSpeed", &dropSpeed, 1.0f);
   }
}

void Patterns::ioParam_dropSpeedRandomMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropSpeed"));
      if (dropSpeed==-1.0f) {
         parent->ioParamValue(ioFlag, name, "dropSpeedRandomMax", &dropSpeedRandomMax, 3.0f);
      }
   }
}

void Patterns::ioParam_dropSpeedRandomMin(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropSpeed"));
      if (dropSpeed==-1.0f) {
         parent->ioParamValue(ioFlag, name, "dropSpeedRandomMin", &dropSpeedRandomMin, 1.0f);
      }
   }
}

void Patterns::ioParam_dropPeriod(enum ParamsIOFlag ioFlag) {
   //(dt) -1 for random period, otherwise, number of frames in between drops
   //TODO: What does dropPeriod of 0 represent? If nothing - why not have 0 be random?
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
       parent->ioParamValue(ioFlag, name, "dropPeriod", &dropPeriod, 1);
   }
}

void Patterns::ioParam_dropPeriodRandomMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropPeriod"));
      if (dropPeriod==-1) {
         parent->ioParamValue(ioFlag, name, "dropPeriodRandomMax", &dropPeriodRandomMax, 20);
      }
   }
}

void Patterns::ioParam_dropPeriodRandomMin(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropPeriod"));
      if (dropPeriod==-1) {
         parent->ioParamValue(ioFlag, name, "dropPeriodRandomMin", &dropPeriodRandomMin, 5);
      }
   }
}

void Patterns::ioParam_dropPosition(enum ParamsIOFlag ioFlag) {
   //Random position is -1 for random number of drops from pos, 0 for drop from center, otherwise
   //number of timesteps in which the drop stays at the position
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
       parent->ioParamValue(ioFlag, name, "dropPosition", &dropPosition, 1);
   }
}

void Patterns::ioParam_dropPositionRandomMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropPosition"));
      if (dropPosition==-1) {
         parent->ioParamValue(ioFlag, name, "dropPositionRandomMax", &dropPositionRandomMax, 20);
      }
   }
}

void Patterns::ioParam_dropPositionRandomMin(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "dropPosition"));
      if (dropPosition==-1) {
         parent->ioParamValue(ioFlag, name, "dropPositionRandomMin", &dropPositionRandomMin, 5);
      }
   }
}

void Patterns::ioParam_halfNeutral(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      parent->ioParamValue(ioFlag, name, "halfNeutral", &onOffFlag, 0);
   }
}

void Patterns::ioParam_minValue(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "halfNeutral"));
      if(onOffFlag){
         parent->ioParamValue(ioFlag, name, "minValue", &minVal, PATTERNS_MINVAL);
      }
      else {
         if (ioFlag==PARAMS_IO_READ) minVal = maxVal;
      }
   }
   else {
      minVal = maxVal;
   }
}

void Patterns::ioParam_inOut(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      parent->ioParamValue(ioFlag, name, "inOut", &inOut, 1.0f);
      //inOut must be between -1 and 1
      if (inOut < -1 || inOut > 1){
         fprintf(stderr, "Patterns:: inOut must be -1 for all in drops, 0 for random, or 1 for all out drops, or anything in between ");
         abort();
      }
   }
}

void Patterns::ioParam_startFrame(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      parent->ioParamValue(ioFlag, name, "startFrame", &startFrame, 0.0);
   }
}

void Patterns::ioParam_endFrame(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "patternType"));
   if (type==DROP) {
      parent->ioParamValue(ioFlag, name, "endFrame", &endFrame, -1.0);
      if (ioFlag == PARAMS_IO_READ && endFrame < 0) {
         endFrame = INT_MAX;
      }
   }
}

void Patterns::ioParam_patternsOutputPath(enum ParamsIOFlag ioFlag) {
   // set output path for movie frames
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeImages"));
   if (writeImages) {
      parent->ioParamString(ioFlag, name, "patternsOutputPath", &patternsOutputPath, parent->getOutputPath());
      if (ioFlag == PARAMS_IO_READ) {
         parent->ensureDirExists(patternsOutputPath);
      }
   }
}

void Patterns::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, 0.0);
}

int Patterns::communicateInitInfo() {
   int status = BaseInput::communicateInitInfo();

   patternRandState = new Random(parent, 1);
#ifndef NDEBUG
   // This should put the RNG into the same state across MPI, but let's check.
   taus_uint4 * state = patternRandState->getRNG(0);
   taus_uint4 checkState;
   memcpy(&checkState, state, sizeof(taus_uint4));
   MPI_Bcast(&checkState, sizeof(taus_uint4), MPI_CHAR, 0, parent->icCommunicator()->communicator());
   assert(!memcmp(state, &checkState, sizeof(taus_uint4)));
#endif // NDEBUG

   const PVLayerLoc * loc = getLayerLoc();
   if(dropPosition == -1){
      nextPosChangeFrame = nextDropFrame + dropPositionRandomMin + floor((dropPositionRandomMax - dropPositionRandomMin) * patternRandState->uniformRandom());
      xPos = (int)floor(loc->nxGlobal * patternRandState->uniformRandom());
      yPos = (int)floor(loc->nyGlobal * patternRandState->uniformRandom());
   }
   else if(dropPosition == 0){
      xPos = (int)floor((loc->nxGlobal - 1) / 2);
      yPos = (int)floor((loc->nyGlobal - 1) / 2);
   }
   else{
      nextPosChangeFrame = nextDropFrame + dropPosition;
      xPos = (int)floor(loc->nxGlobal * patternRandState->uniformRandom());
      yPos = (int)floor(loc->nyGlobal * patternRandState->uniformRandom());
   }
   MPI_Bcast(&nextDropFrame, 1, MPI_DOUBLE, 0, parent->icCommunicator()->communicator());

   return status;
}

int Patterns::allocateDataStructures() {
   int status = BaseInput::allocateDataStructures();
   drawPattern(maxVal);
   return status;
}

int Patterns::tag()
{
   if (orientation == vertical)
      return position;
   else
      return 10*position;
}

int Patterns::drawPattern(float val)
{
   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + loc->halo.lt + loc->halo.rt;
   const int ny = loc->ny + loc->halo.dn + loc->halo.up;

   // reset data buffer
   const int nk = nx * ny;

   float neutralval;
   if (onOffFlag){
      neutralval = .5;
   }
   else{
      neutralval = 0.0;
   }

   for (int k = 0; k < nk; k++) {
      data[k] = neutralval;
   }

   int status = PV_SUCCESS;
   if (type == RECTANGLES) {
      status = drawRectangles(val);
   }
   else if (type == BARS) { // type is bars
      status = drawBars(orientation, data, nx, ny, val);
   }
   else if((type == COSWAVE)||(type == SINEWAVE)||
           (type == COSV)||(type == SINEV)) {
      status = drawWaves(val);
   }
   else if (type == IMPULSE) {
      status = drawImpulse();
   }
   else if (type == DROP){
      status = drawDrops();
   }

   if (normalizeLuminanceFlag) { // Copied from Image::readImage except for names of a couple variables.  Make a normalizeLuminance() function?
      int n=getNumNeurons();
      double image_sum = 0.0f;
      for (int k=0; k<n; k++) {
         image_sum += data[k];
      }
      double image_ave = image_sum / n;
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &image_ave, 1, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      image_ave /= parent->icCommunicator()->commSize();
#endif // PV_USE_MPI
      float image_shift = 0.5f - image_ave;
      for (int k=0; k<n; k++) {
         data[k] += image_shift;
      }
   }

   return 0;
}

int Patterns::drawBars(OrientationMode ormode, pvdata_t * buf, int nx, int ny, float val) {
   int crossstride, alongstride;  // strides in the direction across the bar and along the bar, respectively
   int crosssize, alongsize; // Size of buffer in the direction across the bar and along the bar, respectively
   int wavelength;
   int width;
   switch(ormode) {
   case vertical:
      crossstride = 1; // assumes number of features = 1;
      alongstride = nx;
      crosssize = nx;
      alongsize = ny;
      wavelength = wavelengthVert;
      width = maxWidth;
      break;
   case horizontal:
      crossstride = nx;
      alongstride = 1; // assumes number of features = 1;
      crosssize = ny;
      alongsize = nx;
      wavelength = wavelengthHoriz;
      width = maxHeight;
      break;
   case mixed:
   default:
      assert(0);
      break;
   }

   // data is val on [position,position+width), discretized by pixels
   // Set the pixel at floor(position) to maxVal*( 1-(mod(position,1)) ).
   // Subsequent pixels are set to val (in the while loop), until the
   // remaining mass is less than maxVal;
   // The next pixel after that is set to the remainder.
   // The pattern is then repeated with period wavelength in the direction
   // across the bar; and repeated with period one in the direction along
   // the bar.
   float mass = val*width;
   float point = position;
   int k = ((int) floor(point)) % wavelength;
   float dm = val*(1-(point-k));
   buf[k*crossstride] = dm;
   mass -= dm;
   dm = val;
   while( mass > dm ) {
      k++; k %= wavelength;
      data[k*crossstride] = dm;
      mass -= dm;
   }
   k++; k %= wavelength;
   data[k*crossstride] = mass;

   // Repeat with period wavelength
   for( k = wavelength; k < crosssize; k++ ) {
      data[k*crossstride] = data[(k-wavelength)*crossstride];
   }

   // Repeat in the direction along the bar
   for( int m = 0; m < crosssize; m++ ) {
      int idxsrc = m*crossstride;
      for( k = 1; k < alongsize; k++ ) {
         int idxdest = idxsrc + k*alongstride;
         data[idxdest] = data[idxsrc];
      }
   }

   return PV_SUCCESS;
}

int Patterns::drawRectangles(float val) {
   int status = PV_SUCCESS;
   int width  = (int)(minWidth  + (maxWidth  - minWidth)  * patternRandState->uniformRandom());
   int height = (int)(minHeight + (maxHeight - minHeight) * patternRandState->uniformRandom());
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx + loc->halo.lt + loc->halo.rt;
   const int ny = loc->ny + loc->halo.dn + loc->halo.up;
   const int sx = 1;
   const int sy = sx * nx;

   const int half_w = width/2;
   const int half_h = height/2;

   // random center location
   const int xc = (int)((nx-1) * patternRandState->uniformRandom());
   const int yc = (int)((ny-1) * patternRandState->uniformRandom());

   const int x0 = (xc - half_w < 0) ? 0 : xc - half_w;
   const int y0 = (yc - half_h < 0) ? 0 : yc - half_h;

   const int x1 = (xc + half_w > nx) ? nx : xc + half_w;
   const int y1 = (yc + half_h > ny) ? ny : yc + half_h;

   for (int iy = y0; iy < y1; iy++) {
      for (int ix = x0; ix < x1; ix++) {
         data[ix * sx + iy * sy] = val;
      }
   }
   position = x0 + y0*nx;
   return status;
}

int Patterns::drawWaves(float val) {
   int status = PV_SUCCESS;
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx + loc->halo.lt + loc->halo.rt;
   const int ny = loc->ny + loc->halo.dn + loc->halo.up;
   const int sx = 1;
   const int sy = sx * nx;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nygl = loc->nyGlobal;

   int wavelength=0;
   float rot=0;
   if (orientation == vertical) { // vertical bars
      wavelength = maxWidth;
      rot=rotation;
   }
   else if (orientation == horizontal) { // horizontal bars
      wavelength = maxHeight;
      rot=rotation+PI/2;
   }
   else { // invalid rotation
      assert(true);
   }
   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         int glx = ix+kx0-loc->halo.lt;
         int gly = iy+ky0-loc->halo.up;
         float rot2 = rot;
         float phi = 0;
         if((type == COSV)||(type == SINEV)) {
            float yp=float(glx)*cos(rot) + float(gly)*sin(rot);
            if(yp<nygl/2) {
               rot2+= 3*PI/4;
               phi=float(wavelength)/2;
            }
            else if(yp>nygl/2) {
               rot2+= PI/4;
               phi=0;
            }
         }
         float m = float(glx)*cos(rot2) - float(gly)*sin(rot2) + phi + position; //calculate position including fraction

         //sin of 2*pi*m/wavelength, where wavelength=2*width:
         if((type == SINEWAVE)||(type == SINEV))
            data[ix * sx + iy * sy] = val*sin(PI*m/float(wavelength));
         else if((type == COSWAVE)||(type == COSV))
            data[ix * sx + iy * sy] = val*cos(PI*m/float(wavelength));

      }
   }
   return status;
}

int Patterns::drawImpulse() {
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nxgl = loc->nxGlobal;
   const int nygl = loc->nyGlobal;
   const int sx = 1;
   const int sy = sx * nx;
   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         int glx = ix+kx0-loc->halo.lt;
         int gly = iy+ky0-loc->halo.up;

         if((glx==nxgl/2)&&(gly==nygl/2)&&(initPatternCntr==5))
            data[ix * sx + iy * sy] = 50000.0f;
         else
            data[ix * sx + iy * sy] = 0;
      }
   }
   initPatternCntr++;
   return 0;
}

int Patterns::drawDrops() {
   int status = PV_SUCCESS;
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx + loc->halo.lt + loc->halo.rt;
   const int ny = loc->ny + loc->halo.dn + loc->halo.up;
   const int sx = 1;
   const int sy = sx * nx;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nxgl = loc->nxGlobal;
   const int nygl = loc->nyGlobal;

   // For Drops, jittering moves the centers of the drops by offsetX,offsetY.
   // Since we always redraw the pattern, we don't need jitter's return value.
   // jitter should be added to other pattern types, in which case the call
   // to jitter() should be moved to updatePattern.
   if (jitterFlag) {
      jitter();
      // Because the Patterns layer has its own taus_uint4 random state, and all MPI processes seeded it the same way,
      // all MPI processes have the same offset and bias, without needing to do an MPI call.
   }

   //Max radius at corner of screen
   float max_radius = sqrt(nxgl * nxgl + nygl * nygl);

   //Using iterators to iterate while removing from loop
   for(std::vector<Drop>::iterator dropIt = vDrops.begin(); dropIt < vDrops.end(); dropIt++){
      //Update radius
      dropIt->radius += dropIt->speed;
      //If no longer in the frame either in or out
      if(dropIt->radius >= max_radius || dropIt->radius < 0){

         //Erase from vector, erase returns next iterator object
         dropIt = vDrops.erase(dropIt);
      }
   }

   //Change x and y position if needed
   if(framenumber >= nextPosChangeFrame && dropPosition != 0){
      if(dropPosition == -1){
         nextPosChangeFrame += dropPositionRandomMin + floor((dropPositionRandomMax - dropPositionRandomMin) * patternRandState->uniformRandom());
         xPos = (int)(floor(loc->nxGlobal * patternRandState->uniformRandom()));
         yPos = (int)(floor(loc->nyGlobal * patternRandState->uniformRandom()));
      }
      else{
         nextPosChangeFrame += dropPosition;
         xPos = (int)(floor(loc->nxGlobal * patternRandState->uniformRandom()));
         yPos = (int)(floor(loc->nyGlobal * patternRandState->uniformRandom()));
      }
      //No need to communicate it since drop creator will decide where to drop
   }

   //Add new circles
   if(framenumber >= nextDropFrame && framenumber <= endFrame){
      if(dropPeriod == -1){
         nextDropFrame = framenumber + dropPeriodRandomMin + floor((dropPeriodRandomMax - dropPeriodRandomMin) * patternRandState->uniformRandom());
      }
      else{
         nextDropFrame = framenumber + dropPeriod;
      }
      //Create new structure
      Drop newDrop;
      //Random drop speed
      if(dropSpeed == -1){
         newDrop.speed = dropSpeedRandomMin + (dropSpeedRandomMax - dropSpeedRandomMin) * patternRandState->uniformRandom();
      }
      else{
         newDrop.speed = dropSpeed;
      }
      newDrop.centerX = xPos;
      newDrop.centerY = yPos;
      //Random on/off input
      if(patternRandState->uniformRandom() < .5){
         newDrop.on = true;
      }
      else{
         newDrop.on = false;
      }
      memset(&newDrop.padding, 0, 3);
      //If out is true, out drop
      bool out;
      //Convert inOut into a scale between 0 and 1, where 0 maps to in and 1 maps to out
      float inOutChance = (inOut + 1)/2;
      if(patternRandState->uniformRandom() < inOutChance){
         out = true;
      }
      else{
         out = false;
      }

      if (out){
         newDrop.radius = 0;
      }
      else{
         //Start at max radius
         newDrop.radius = max_radius;
         //Reverse sign of speed
         newDrop.speed = -newDrop.speed;
      }

      //Communicate to rest of processors
      MPI_Bcast(&nextDropFrame, 1, MPI_DOUBLE, 0, parent->icCommunicator()->communicator());
      MPI_Bcast(&newDrop, sizeof(Drop), MPI_BYTE, 0, parent->icCommunicator()->communicator());
      vDrops.push_back(newDrop);
   }

   //Draw circle
   for(int i = 0; i < (int)vDrops.size(); i++){
      double delta_theta = fabs(atan(0.1/(double) vDrops[i].radius));
      for (double theta = 0; theta < 2*PI; theta += delta_theta){
         int ix = (int)(round(getOffsetX(this->offsetAnchor, this->offsets[0]) + vDrops[i].centerX + vDrops[i].radius * cos(theta)));
         int iy = (int)(round(getOffsetY(this->offsetAnchor, this->offsets[1]) + vDrops[i].centerY + vDrops[i].radius * sin(theta)));
         //Check edge bounds based on nx/ny size
         if(ix < nx + kx0 && iy < ny + ky0 && ix >= kx0 && iy >= ky0){
            //Random either on circle or off circle
            if(vDrops[i].on){
               data[(ix - kx0) * sx + (iy - ky0) * sy] = maxVal;
            }
            else{
               data[(ix - kx0) * sx + (iy - ky0) * sy] = minVal;
            }
         }
      }
   }//End radius for loop
   return status;
}

//Image never updates, so getDeltaUpdateTime should return update on every timestep
//TODO see when this layer actually needs to update
double Patterns::getDeltaUpdateTime(){
   return displayPeriod;
}

//bool Patterns::needUpdate(double timef, double dt){
//   framenumber = timef * dt;
//   bool needNewPattern = timef >= nextDisplayTime;
//   if (needNewPattern) {
//      nextDisplayTime += displayPeriod;
//   }
//   return needNewPattern;
//}


/**
 * update the image buffers
 */
int Patterns::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   //Moved to needUpdate()
   //framenumber = timef * dt;
   //bool needNewPattern = timef >= nextDisplayTime;
   //if (needNewPattern) {

   //   nextDisplayTime += displayPeriod;
   status = getFrame(timef, dt);
   //}
   return status;
}

//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int Patterns::retrieveData(double timef, double dt)
{
   int status = PV_SUCCESS;
   status = updatePattern(timef);
   return status;
}

int Patterns::updatePattern(double timef) {
   //update_timer->start();

   // alternate between vertical and horizontal bars
   double p = patternRandState->uniformRandom();
   bool newPattern = false;


   if (type==RECTANGLES || type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      if( p < pSwitch) { // switch orientation with probability pSwitch
         setOrientation(orientation == vertical ? horizontal : vertical);
         newPattern = true;
      }
      // moving probability
      p -= pSwitch; // Doesn't make sense to both switch and move
      if (p >= 0 && p < pMove) {
         newPattern = true;
      }
      if (newPattern) {
         position = calcPosition(position, positionBound);
      }
   }

   if (type == DROP){
      newPattern = true;
   }

   if (newPattern) {
      //lastUpdateTime = timef;

      drawPattern(maxVal);
      if (writeImages) {
         char basicfilename[PV_PATH_MAX];
         if (type == BARS)
            snprintf(basicfilename, PV_PATH_MAX, "%s/Bars_%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         else if (type == RECTANGLES){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Rectangles_%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == SINEWAVE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Sinewave%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == COSWAVE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Coswave%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == SINEV){
            snprintf(basicfilename, PV_PATH_MAX, "%s/SineV%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == COSV){
            snprintf(basicfilename, PV_PATH_MAX, "%s/CosV%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == IMPULSE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Impulse%.2f.%s", patternsOutputPath, timef, writeImagesExtension);
         }
         else if (type == DROP){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Drop%.3d.%s", patternsOutputPath, (int)timef, writeImagesExtension);
         }
         writeImage(basicfilename, 0); //TODO incorporate batch idx here
      }
   }

   //update_timer->stop();

   return (int) newPattern;
}

/**
 *
 *  Return a value in the interval [0,step)
 *  For movementType RANDOMJUMP or RANDOMWALK, value is integral
 *  For MOVEFORWARD or MOVEBACKWARD, returns a float
 */
float Patterns::calcPosition(float pos, int step)
{
   double p = patternRandState->uniformRandom();

   switch (movementType) {
   case MOVEFORWARD:
     pos = (pos+movementSpeed) ;
     if(pos>step) {pos -= step;}
     if(pos<0) {pos += step;}
     if(pos>step) {pos -= step;}
    break;
   case MOVEBACKWARD:
      pos = (pos-movementSpeed) ;
      if(pos<0) {pos += step;}
      if(pos>step) {pos -= step;}
     break;
   case RANDOMJUMP:
      pos = floor(p * step);
      break;
   case RANDOMWALK:
   default: //in case of any problems with setting the movementType var, just use the
            //random walk as default
      if (p < 0.5){
         pos = ((int)(pos+movementSpeed)) % step;
      } else {
         pos = ((int)(pos-movementSpeed)) % step;
      }
      break;
   }
   if (patternsFile != NULL) {
      assert(parent->columnId()==0);
      fprintf(patternsFile->fp, "Time %f, position %f\n", parent->simulationTime(), pos);
   }

   return pos;
}

int Patterns::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = BaseInput::readStateFromCheckpoint(cpDir, timeptr);
   status = readPatternStateFromCheckpoint(cpDir);
   return status;
}

int Patterns::readPatternStateFromCheckpoint(const char * cpDir) {
   int status = PV_SUCCESS;
   if( parent->columnId() == 0 ) {
      char * filename = parent->pathInCheckpoint(cpDir, getName(), "_PatternState.bin");
      PV_Stream * pvstream = PV_fopen(filename, "r", false/*verifyWrites*/);
      if( pvstream != NULL ) {
         status = PV_fread(&type, sizeof(PatternType), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fread(&patternRandState, sizeof(taus_uint4), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fread(&orientation, sizeof(OrientationMode), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fread(&position, sizeof(float), 1, pvstream) ? status : PV_FAILURE;
         //status = PV_fread(&nextDisplayTime, sizeof(double), 1, pvstream) ? status : PV_FAILURE;
         status = PV_fread(&nextDropFrame, sizeof(double), 1, pvstream) ? status : PV_FAILURE;
         status = PV_fread(&nextPosChangeFrame, sizeof(double), 1, pvstream) ? status : PV_FAILURE;
         status = PV_fread(&initPatternCntr, sizeof(int), 1, pvstream) ? status : PV_FAILURE;
         status = PV_fread(&xPos, sizeof(int), 1, pvstream) ? status : PV_FAILURE;
         status = PV_fread(&yPos, sizeof(int), 1, pvstream) ? status : PV_FAILURE;
         int size;
         status = PV_fread(&size, sizeof(int), 1, pvstream) ? status : PV_FAILURE;
         vDrops.clear();
         for (int k=0; k<size; k++) {
            Drop drop;
            PV_fread(&drop, sizeof(Drop), 1, pvstream);
            vDrops.push_back(drop);
         }
         assert((int)vDrops.size()==size);
         PV_fclose(pvstream);
      }
      else {
         fprintf(stderr, "Unable to read from \"%s\"\n", filename);
      }
   }

   // TODO improve and polish the way the code handles file I/O and the MPI data buffer.
   // This will get bad if the number of member variables that need to be saved keeps increasing.
#ifdef PV_USE_MPI
   if (parent->icCommunicator()->commSize()>1) {
      int bufsize = (int) (sizeof(PatternType) + sizeof(taus_uint4) + sizeof(OrientationMode) + 1*sizeof(float) + 2*sizeof(double) + 4*sizeof(int) + vDrops.size()*sizeof(Drop));
      //Communicate buffer size to rest of processes
      MPI_Bcast(&bufsize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      char tempbuf[bufsize];
      PatternType * savedtype = (PatternType *) (tempbuf+0);
      taus_uint4 * rstate = (taus_uint4 *) (tempbuf+sizeof(PatternType));
      OrientationMode * om = (OrientationMode *) (tempbuf+sizeof(PatternType)+sizeof(taus_uint4));
      float * floats = (float *) (tempbuf+sizeof(PatternType)+sizeof(taus_uint4)+sizeof(OrientationMode));
      double * doubles = (double *) (tempbuf+sizeof(PatternType)+sizeof(taus_uint4)+sizeof(OrientationMode)+sizeof(float));
      int * ints = (int *) (tempbuf+sizeof(PatternType)+sizeof(taus_uint4)+sizeof(OrientationMode)+sizeof(float)+3*sizeof(double));
      Drop * drops = (Drop *) (tempbuf+sizeof(PatternType)+sizeof(taus_uint4)+sizeof(OrientationMode)+sizeof(float)+3*sizeof(double)+4*sizeof(int));
      int numdrops;
      if (parent->columnId()==0) {
         *savedtype = type;
         memcpy(rstate, patternRandState->getRNG(0), sizeof(taus_uint4));
         *om = orientation;
         floats[0] = position;
         //doubles[0] = nextDisplayTime;
         doubles[1] = nextDropFrame;
         doubles[2] = nextPosChangeFrame;
         ints[0] = initPatternCntr;
         ints[1] = xPos;
         ints[2] = yPos;
         numdrops = (int) vDrops.size();
         ints[3] = numdrops;
         for (int k=0; k<numdrops; k++) {
            memcpy(&(drops[k]), &(vDrops[k]), sizeof(Drop));
         }
         MPI_Bcast(tempbuf, bufsize, MPI_CHAR, 0, parent->icCommunicator()->communicator());
      }
      else {
         MPI_Bcast(tempbuf, bufsize, MPI_CHAR, 0, parent->icCommunicator()->communicator());
         type = *savedtype;
         memcpy(patternRandState->getRNG(0), rstate, sizeof(taus_uint4));
         orientation = *om;
         position = floats[0];
         //nextDisplayTime = doubles[0];
         nextDropFrame = doubles[1];
         nextPosChangeFrame = doubles[2];
         initPatternCntr = ints[0];
         xPos = ints[1];
         yPos = ints[2];
         numdrops = ints[3];
         vDrops.clear();
         for (int k=0; k<numdrops; k++) {
            Drop drop = drops[k];
            vDrops.push_back(drop);
         }
         assert((int)vDrops.size()==numdrops);
      }
   }
#endif // PV_USE_MPI
   //free(filename);
   return status;
}

int Patterns::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   int filenamesize = strlen(cpDir)+1+strlen(name)+18;
   // The +1 is for the slash between cpDir and name; the +18 needs to be large enough to hold the suffix _PatternState.{bin,txt} plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);

   sprintf(filename, "%s/%s_PatternState.bin", cpDir, name);
   if( icComm->commRank() == 0 ) {
      PV_Stream * pvstream = PV_fopen(filename, "w", parent->getVerifyWrites());
      int size = vDrops.size();
      if( pvstream != NULL ) {
         status = PV_fwrite(&type, sizeof(PatternType), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&patternRandState, sizeof(taus_uint4), 1, pvstream) == 1 ? status : PV_FAILURE;
         // This should only write the variables used by PatternType type.
         // For example, DROP doesn't use orientation and BARS doesn't use nextDropFrame
         status = PV_fwrite(&orientation, sizeof(OrientationMode), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&position, sizeof(float), 1, pvstream) == 1 ? status : PV_FAILURE;
         //status = PV_fwrite(&nextDisplayTime, sizeof(double), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&nextDropFrame, sizeof(double), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&nextPosChangeFrame, sizeof(double), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&initPatternCntr, sizeof(int), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&xPos, sizeof(int), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&yPos, sizeof(int), 1, pvstream) == 1 ? status : PV_FAILURE;
         status = PV_fwrite(&size, sizeof(int), 1, pvstream) == 1 ? status : PV_FAILURE;
         for (int k=0; k<size; k++) {
            status = PV_fwrite(&vDrops[k], sizeof(Drop), 1, pvstream) == 1 ? status : PV_FAILURE;
         }
         PV_fclose(pvstream);
      }
      else {
         fprintf(stderr, "Unable to write to \"%s\"\n", filename);
      }
      if (status != PV_SUCCESS) {
         fprintf(stderr, "Patterns::checkpointWrite error: %s \"%s\" failed writing to %s\n", getKeyword(), name, pvstream->name);
         exit(EXIT_FAILURE);
      }
      sprintf(filename, "%s/%s_PatternState.txt", cpDir, name);
      pvstream = PV_fopen(filename, "w", parent->getVerifyWrites());
      fprintf(pvstream->fp, "Orientation = ");
      switch(orientation) {
      case horizontal:
         fprintf(pvstream->fp, "horizontal\n");
         break;
      case vertical:
         fprintf(pvstream->fp, "vertical\n");
         break;
      case mixed:
         fprintf(pvstream->fp, "mixed\n");
         break;
      default:
         assert(0);
         break;
      }
      switch(type) {
      case BARS:
         fprintf(pvstream->fp, "Type = %d (BARS)\n", type);
         break;
      case RECTANGLES:
         fprintf(pvstream->fp, "Type = %d (RECTANGLES)\n", type);
         break;
      case SINEWAVE:
         fprintf(pvstream->fp, "Type = %d (SINEWAVE)\n", type);
         break;
      case COSWAVE:
         fprintf(pvstream->fp, "Type = %d (COSWAVE)\n", type);
         break;
      case IMPULSE:
         fprintf(pvstream->fp, "Type = %d (IMPULSE)\n", type);
         break;
      case SINEV:
         fprintf(pvstream->fp, "Type = %d (SINEV)\n", type);
         break;
      case COSV:
         fprintf(pvstream->fp, "Type = %d (COSV)\n", type);
         break;
      case DROP:
         fprintf(pvstream->fp, "Type = %d (DROP)\n", type);
         break;
      }
      taus_uint4 * rng = patternRandState->getRNG(0);
      fprintf(pvstream->fp, "Random state = %u, %u, %u, %u\n", rng->s0, rng->state.s1, rng->state.s2, rng->state.s3);
      fprintf(pvstream->fp, "Position = %f\n", position);
      //fprintf(pvstream->fp, "nextDisplayTime = %f\n", nextDisplayTime);
      fprintf(pvstream->fp, "initPatternCntr = %d\n", initPatternCntr);
      fprintf(pvstream->fp, "nextDropFrame = %f\n", nextDropFrame);
      fprintf(pvstream->fp, "nextPosChangeFrame = %f\n", nextPosChangeFrame);
      fprintf(pvstream->fp, "xPos = %d\n", xPos);
      fprintf(pvstream->fp, "yPos = %d\n", yPos);
      fprintf(pvstream->fp, "size of vDrops vector = %d\n", size);
      for (int k=0; k<size; k++) {
         fprintf(pvstream->fp, "   element %d: centerX = %d, centerY = %d, speed = %f, radius = %f, on = %d\n",
                     k, vDrops[k].centerX, vDrops[k].centerY, vDrops[k].speed, vDrops[k].radius, vDrops[k].on);
      }
      PV_fclose(pvstream);
   }
   free(filename); filename=NULL;
   return PV_SUCCESS;
}

BaseObject * createPatterns(char const * name, HyPerCol * hc) {
   return hc ? new Patterns(name, hc) : NULL;
}

} // namespace PV
