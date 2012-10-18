/*
 * Patterns.cpp
 *
 *  Created on: April 21, 2010
 *      Author: Marian Anghel and Craig Rasmussen
 */

#include "Patterns.hpp"
#include "../include/pv_common.h"  // for PI
#include "../utils/pv_random.h"

#define PATTERNS_MAXVAL  1.0f
#define PATTERNS_MINVAL  0.0f

namespace PV {

// CER-new

Patterns::Patterns() {
   initialize_base();
}

Patterns::Patterns(const char * name, HyPerCol * hc, PatternType type) {
   initialize_base();
   initialize(name, hc, type);
}

Patterns::~Patterns()
{
   free(patternsOutputPath);
   vDrops.clear();

   if( patternsFile != NULL ) {
      fclose(patternsFile);
      patternsFile = NULL;
   }
}

int Patterns::initialize_base() {
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

int Patterns::initialize(const char * name, HyPerCol * hc, PatternType type) {
   Image::initialize(name, hc, NULL);
   assert(getLayerLoc()->nf == 1);
   this->type = type;
   PVParams * params = hc->parameters();
   const PVLayerLoc * loc = getLayerLoc();

   if (type==BARS) {
      orientation = readOrientation();
      setOrientation(orientation);
      pMove   = params->value(name, "pMove", 0.0);
      pSwitch = params->value(name, "pSwitch", 0.0);
      movementType = readMovementType();
      movementSpeed = params->value(name, "movementSpeed", 1);
      writePosition = (int) params->value(name,"writePosition", 0);
      maxVal = params->value(name,"maxValue", PATTERNS_MAXVAL);
      maxWidth  = params->value(name, "width", loc->nx); // width of bar when bar is vertical
      maxHeight = params->value(name, "height", loc->ny); // height of bar when bar is horizontal
      wavelengthVert = params->value(name, "wavelengthVert", 2*maxWidth);
      wavelengthHoriz = params->value(name, "wavelengthHoriz", 2*maxHeight);
      position = 0;
   }
   if (type==RECTANGLES) {
      maxVal = params->value(name,"maxValue", PATTERNS_MAXVAL);
      maxWidth  = params->value(name, "maxWidth", loc->nx);
      maxHeight = params->value(name, "maxHeight", loc->ny);
      minWidth = params->value(name, "minWidth", maxWidth);
      minHeight = params->value(name, "minWeight", maxHeight);
      movementType = readMovementType();
      movementSpeed = params->value(name, "movementSpeed", 1);
      writePosition = (int) params->value(name,"writePosition", 0);
   }
   if (type==SINEWAVE || type==COSWAVE || type==SINEV || type==COSV) {
      orientation = readOrientation();
      setOrientation(orientation);
      pMove   = params->value(name, "pMove", 0.0);
      pSwitch = params->value(name, "pSwitch", 0.0);
      maxVal = params->value(name,"maxValue", PATTERNS_MAXVAL);
      maxWidth  = params->value(name, "width", loc->nx); // width of bar when bar is vertical
      maxHeight = params->value(name, "height", loc->ny); // height of bar when bar is horizontal
      wavelengthVert = params->value(name, "wavelengthVert", 2*maxWidth);
      wavelengthHoriz = params->value(name, "wavelengthHoriz", 2*maxHeight);
      rotation = params->value(name, "rotation", 0.0);
      writePosition = (int) params->value(name,"writePosition", 0);
   }
   if (type==IMPULSE) {
      initPatternCntr = 0;
   }
   if (type==DROP) {
      dropSpeed = params->value(name, "dropSpeed", 1);
      dropSpeedRandomMax = params->value(name, "dropSpeedRandomMax", 3);
      dropSpeedRandomMin = params->value(name, "dropSpeedRandomMin", 1);

      dropPeriod = params->value(name, "dropPeriod", 10);
      dropPeriodRandomMax = params->value(name, "dropPeriodRandomMax", 20);
      dropPeriodRandomMin = params->value(name, "dropPeriodRandomMin", 5);

      //Random position is -1 for random number of drops from pos, 0 for drop from center, otherwise
      //number of timesteps in which the drop stays at the position
      dropPosition = params->value(name, "dropPosition", 0);
      dropPositionRandomMax = params->value(name, "dropPositionRandomMax", 20);
      dropPositionRandomMin = params->value(name, "dropPositionRandomMin", 5);

      maxVal = params->value(name,"maxValue", PATTERNS_MAXVAL);
      onOffFlag = params->value(name, "halfNeutral", 0);

      if(onOffFlag){
         minVal = params->value(name,"minValue", PATTERNS_MINVAL);
      }
      else {
         minVal = maxVal;
      }

      startFrame = params->value(name, "startFrame", 0);
      endFrame = params->value(name, "endFrame", -1);
      if (endFrame < 0) endFrame = INT_MAX;
      //Assign first drop
      //radius.push_back(0);
      //Assign next drop
      if(dropPeriod == -1){
         nextDropFrame = ceil(startFrame);
      }
      else{
         nextDropFrame = dropPeriod;
      }

      if(dropPosition == -1){
         nextPosChangeFrame = nextDropFrame + dropPositionRandomMin + (dropPositionRandomMax - dropPositionRandomMin) * pv_random_prob();
         xPos = floor(loc->nxGlobal * pv_random_prob());
         yPos = floor(loc->nyGlobal * pv_random_prob());
      }
      else if(dropPosition == 0){
         xPos = floor((loc->nxGlobal - 1) / 2);
         yPos = floor((loc->nyGlobal - 1) / 2);
      }
      else{
         nextPosChangeFrame = nextDropFrame + dropPosition;
         xPos = floor(loc->nxGlobal * pv_random_prob());
         yPos = floor(loc->nyGlobal * pv_random_prob());
      }
      MPI_Bcast(&nextDropFrame, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
   }

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);
   // set output path for movie frames
   if(writeImages){
      if ( params->stringPresent(name, "patternsOutputPath") ) {
         patternsOutputPath = strdup(params->stringValue(name, "patternsOutputPath"));
         assert(patternsOutputPath != NULL);
         hc->ensureDirExists(patternsOutputPath);
      }
      else {
         patternsOutputPath = strdup( hc->getOutputPath());
         assert(patternsOutputPath != NULL);
         printf("patternsOutputPath is not specified in params file.\n"
               "Patterns output path set to default \"%s\"\n",patternsOutputPath);
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
         patternsFile = fopen(file_name,"a");
         if(patternsFile == NULL) {
            fprintf(stderr, "Patterns layer \"%s\" unable to open \"%s\" for writing: error %s\n", name, file_name, strerror(errno));
            abort();
         }
      }
      else {
         patternsFile = NULL; // Only root process should write to patternsFile
      }
   }


   displayPeriod = params->value(name,"displayPeriod", 0.0f);
   // displayPeriod = 0 means nextDisplayTime will always >= starting time and therefore the pattern will update every timestep
   nextDisplayTime = hc->simulationTime() + displayPeriod;

   drawPattern(maxVal);

   return PV_SUCCESS;
}

OrientationMode Patterns::readOrientation() {
   const char * allowedOrientationModes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "HORIZONTAL",
         "VERTICAL",
         "MIXED",
         "_End_allowedOrientationTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   OrientationMode ormode = vertical;
   //if the orientation isn't set, use vertical as the default...
   const char * orientationModeStr = parent->parameters()->stringValue(name, "orientation");
   if( ! orientationModeStr ) {
      ormode = vertical; // PVParams::stringValue will print a warning message if absent
   }
   else {
      int orientationModeMatch = false;
      for( int i=0; strcmp(allowedOrientationModes[i],"_End_allowedOrientationTypes"); i++ ) {
         const char * thisorientationmode = allowedOrientationModes[i];
         if( !strcmp(orientationModeStr, thisorientationmode) ) {
            ormode = (OrientationMode) i;
            orientationModeMatch = true;
            break;
         }
      }
      if( !orientationModeMatch ) { //if the set orientation isn't recognized, use vertical as default
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: orientation mode \"%s\" not recognized.  Using VERTICAL.\n", orientationModeStr);
         }
         ormode = vertical;
      }
   }
   return ormode;
}

int Patterns::setOrientation(OrientationMode ormode) {
   orientation = ormode;
   switch(ormode) {
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

MovementType Patterns::readMovementType() {
   //set movement type (random walk is default)
   MovementType movement_type = RANDOMWALK;
   const char * allowedMovementTypes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "RANDOMWALK",
         "MOVEFORWARD",
         "MOVEBACKWARD",
         "RANDOMJUMP",
         "_End_allowedPatternTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   //if the movement type isn't set, use random walk as the default...
   const char * movementTypeStr = parent->parameters()->stringValue(name, "movementType");
   if( ! movementTypeStr ) {
      movement_type = RANDOMWALK; // PVParams::stringValue will print a warning message if absent
   }
   else {
      int movementTypeMatch = false;
      for( int i=0; strcmp(allowedMovementTypes[i],"_End_allowedPatternTypes"); i++ ) {
         const char * thisMovementType = allowedMovementTypes[i];
         if( !strcmp(movementTypeStr, thisMovementType) ) {
            movement_type = (MovementType) i;
            movementTypeMatch = true;
            break;
         }
      }
      if( !movementTypeMatch ) { //if the set movement type isn't recognized, use random walk as default
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: movement type \"%s\" not recognized.  Using RANDOMWALK.\n", movementTypeStr);
         }
         movement_type = RANDOMWALK;
      }
   }
   return movement_type;
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

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;

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

   if (type == RECTANGLES) {
      return drawRectangles(val);
   }
   else if (type == BARS) { // type is bars
      return drawBars(orientation, data, nx, ny, val);
   }
   else if((type == COSWAVE)||(type == SINEWAVE)||
           (type == COSV)||(type == SINEV)) {
      return drawWaves(val);
   }
   else if (type == IMPULSE) {
      return drawImpulse();
   }
   else if (type == DROP){
      return drawDrops();
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
   int width  = minWidth  + (maxWidth  - minWidth)  * pv_random_prob();
   int height = minHeight + (maxHeight - minHeight) * pv_random_prob();
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int sx = 1;
   const int sy = sx * nx;

   const int half_w = width/2;
   const int half_h = height/2;

   // random center location
   const int xc = (nx-1) * pv_random_prob();
   const int yc = (ny-1) * pv_random_prob();

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
   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int nb = loc->nb;
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
         int glx = ix+kx0-nb;
         int gly = iy+ky0-nb;
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
   const int nb = loc->nb;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nxgl = loc->nxGlobal;
   const int nygl = loc->nyGlobal;
   const int sx = 1;
   const int sy = sx * nx;
   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         int glx = ix+kx0-nb;
         int gly = iy+ky0-nb;

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
   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int sx = 1;
   const int sy = sx * nx;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nxgl = loc->nxGlobal;
   const int nygl = loc->nyGlobal;

   //Max radius at corner of screen
   float max_radius = sqrt(nxgl * nxgl + nygl * nygl);

   //Using iterators to iterate while removing from loop
   for(std::vector<Drop>::iterator dropIt = vDrops.begin(); dropIt < vDrops.end(); dropIt++){
      //Update radius
      dropIt->radius += dropIt->speed;
      //If no longer in the frame
      if(dropIt->radius >= max_radius){

         //Erase from vector, erase returns next iterator object
         dropIt = vDrops.erase(dropIt);
      }
   }

   //Change x and y position if needed
   if(framenumber >= nextPosChangeFrame && dropPosition != 0){
      if(dropPosition == -1){
         nextPosChangeFrame += dropPositionRandomMin + (dropPositionRandomMax - dropPositionRandomMin) * pv_random_prob();
         xPos = floor(loc->nxGlobal * pv_random_prob());
         yPos = floor(loc->nyGlobal * pv_random_prob());
      }
      else{
         nextPosChangeFrame += dropPosition;
         xPos = floor(loc->nxGlobal * pv_random_prob());
         yPos = floor(loc->nyGlobal * pv_random_prob());
      }
      //No need to communicate it since drop creator will decide where to drop
   }

   //Add new circles
   if(framenumber >= nextDropFrame && framenumber <= endFrame){
      if(dropPeriod == -1){
         nextDropFrame = framenumber + dropPeriodRandomMin + floor((dropPeriodRandomMax - dropPeriodRandomMin) * pv_random_prob());
      }
      else{
         nextDropFrame = framenumber + dropPeriod;
      }
      //Create new structure
      Drop newDrop;
      //Random drop speed
      if(dropSpeed == -1){
         newDrop.speed = dropSpeedRandomMin + floor((dropSpeedRandomMax - dropSpeedRandomMin) * pv_random_prob());
      }
      else{
         newDrop.speed = dropSpeed;
      }
      newDrop.centerX = xPos;
      newDrop.centerY = yPos;
      //Random on/off input
      if(pv_random_prob() < .5){
         newDrop.on = true;
      }
      else{
         newDrop.on = false;
      }
      newDrop.radius = 0;

      //Communicate to rest of processors
      MPI_Bcast(&nextDropFrame, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      MPI_Bcast(&newDrop, sizeof(Drop), MPI_BYTE, 0, parent->icCommunicator()->communicator());
      vDrops.push_back(newDrop);
   }

   //Draw circle
   for(int i = 0; i < (int)vDrops.size(); i++){
      float delta_theta = fabs(atan((float)1./vDrops[i].radius));
      for (float theta = 0; theta < 2*PI; theta += delta_theta){
        // std::cout << "\t" << theta << "\n";
         int ix = round(vDrops[i].centerX + vDrops[i].radius * cos(theta));
         int iy = round(vDrops[i].centerY + vDrops[i].radius * sin(theta));

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

/**
 * update the image buffers
 */
int Patterns::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
   framenumber = timef * dt;
   bool needNewPattern = timef >= nextDisplayTime;
   if (needNewPattern) {

      nextDisplayTime += displayPeriod;
      status = updatePattern(timef);
   }
   return status;
}

int Patterns::updatePattern(float timef) {
   update_timer->start();

   // alternate between vertical and horizontal bars
   double p = pv_random_prob();
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
      lastUpdateTime = timef;

      drawPattern(maxVal);
      if (writeImages) {
         char basicfilename[PV_PATH_MAX];
         if (type == BARS)
            snprintf(basicfilename, PV_PATH_MAX, "%s/Bars_%.2f.tif", patternsOutputPath, timef);
         else if (type == RECTANGLES){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Rectangles_%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == SINEWAVE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Sinewave%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == COSWAVE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Coswave%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == SINEV){
            snprintf(basicfilename, PV_PATH_MAX, "%s/SineV%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == COSV){
            snprintf(basicfilename, PV_PATH_MAX, "%s/CosV%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == IMPULSE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Impulse%.2f.tif", patternsOutputPath, timef);
         }
         else if (type == DROP){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Drop%.3d.tif", patternsOutputPath, (int)timef);
         }
         write(basicfilename);
      }
   }

   update_timer->stop();

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
   double p = pv_random_prob();

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
      fprintf(patternsFile, "Time %f, position %f\n", parent->simulationTime(), pos);
   }

   return pos;
}


int Patterns::checkpointRead(const char * cpDir, float * timef) {
   int status = HyPerLayer::checkpointRead(cpDir, timef);
   InterColComm * icComm = parent->icCommunicator();
   int filenamesize = strlen(cpDir)+1+strlen(name)+18;
   // The +1 is for the slash between cpDir and name; the +18 needs to be large enough to hold the suffix _PatternState.{bin,txt} plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);

   int chars_needed = snprintf(filename, filenamesize, "%s/%s_PatternState.bin", cpDir, name);
   assert(chars_needed < filenamesize);
   if( icComm->commRank() == 0 ) {
      FILE * fp = fopen(filename, "r");
      if( fp != NULL ) {
         status = fread(&orientation, sizeof(OrientationMode), 1, fp) == 1 ? status : PV_FAILURE;
         status = fread(&position, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fread(&nextDisplayTime, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fread(&initPatternCntr, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fread(&nextDropFrame, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fread(&nextPosChangeFrame, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fread(&xPos, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fread(&yPos, sizeof(int), 1, fp) ? status : PV_FAILURE;
         int size;
         status = fread(&size, sizeof(int), 1, fp) ? status : PV_FAILURE;
         vDrops.clear();
         for (int k=0; k<size; k++) {
            Drop drop;
            fread(&drop, sizeof(Drop), 1, fp);
            vDrops.push_back(drop);
         }
         assert((int)vDrops.size()==size);
         fclose(fp);
      }
      else {
         fprintf(stderr, "Unable to read from \"%s\"\n", filename);
      }
   }

   // TODO improve and polish the way the code handles file I/O and the MPI data buffer.
   // This will get bad if the number of member variables that need to be saved keeps increasing.
#ifdef PV_USE_MPI
   if (parent->icCommunicator()->commSize()>1) {
      int bufsize = sizeof(OrientationMode) + 2*sizeof(float) + 6*sizeof(int) + vDrops.size()*sizeof(Drop);
      //Communicate buffer size to rest of processes
      MPI_Bcast(&bufsize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      char tempbuf[bufsize];
      OrientationMode * om = (OrientationMode *) (tempbuf+0);
      float * floats = (float *) (tempbuf+sizeof(OrientationMode));
      int * ints = (int *) (tempbuf+sizeof(OrientationMode)+2*sizeof(float));
      Drop * drops = (Drop *) (tempbuf+sizeof(OrientationMode)+2*sizeof(float)+6*sizeof(int));
      int numdrops;
      if (parent->columnId()==0) {
         *om = orientation;
         floats[0] = position;
         floats[1] = nextDisplayTime;
         ints[0] = initPatternCntr;
         ints[1] = nextDropFrame;
         ints[2] = nextPosChangeFrame;
         ints[3] = xPos;
         ints[4] = yPos;
         numdrops = (int) vDrops.size();
         ints[5] = numdrops;
         for (int k=0; k<numdrops; k++) {
            memcpy(&(drops[k]), &(vDrops[k]), sizeof(Drop));
         }
         MPI_Bcast(tempbuf, bufsize, MPI_CHAR, 0, parent->icCommunicator()->communicator());
      }
      else {
         MPI_Bcast(tempbuf, bufsize, MPI_CHAR, 0, parent->icCommunicator()->communicator());
         orientation = *om;
         position = floats[0];
         nextDisplayTime = floats[1];
         initPatternCntr = ints[0];
         nextDropFrame = ints[1];
         nextPosChangeFrame = ints[2];
         xPos = ints[3];
         yPos = ints[4];
         numdrops = ints[5];
         vDrops.clear();
         for (int k=0; k<numdrops; k++) {
            Drop drop = drops[k];
            vDrops.push_back(drop);
         }
         assert((int)vDrops.size()==numdrops);
      }
   }
#endif // PV_USE_MPI
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
      FILE * fp = fopen(filename, "w");
      int size = vDrops.size();
      if( fp != NULL ) {
         status = fwrite(&orientation, sizeof(OrientationMode), 1, fp) == 1 ? status : PV_FAILURE;
         status = fwrite(&position, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&nextDisplayTime, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&initPatternCntr, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&nextDropFrame, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&nextPosChangeFrame, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&xPos, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&yPos, sizeof(int), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&size, sizeof(int), 1, fp) ? status : PV_FAILURE;
         for (int k=0; k<size; k++) {
            status = fwrite(&vDrops[k], sizeof(Drop), 1, fp) ? status : PV_FAILURE;
         }
         fclose(fp);
      }
      else {
         fprintf(stderr, "Unable to write to \"%s\"\n", filename);
      }
      sprintf(filename, "%s/%s_PatternState.txt", cpDir, name);
      fp = fopen(filename, "w");
      fprintf(fp, "Orientation = ");
      switch(orientation) {
      case horizontal:
         fprintf(fp, "horizontal\n");
         break;
      case vertical:
         fprintf(fp, "vertical\n");
         break;
      case mixed:
         fprintf(fp, "mixed\n");
         break;
      default:
         assert(0);
         break;
      }
      fprintf(fp, "Position = %f\n", position);
      fprintf(fp, "nextDisplayTime = %f\n", nextDisplayTime);
      fprintf(fp, "initPatternCntr = %d\n", initPatternCntr);
      fprintf(fp, "nextDropFrame = %d\n", nextDropFrame);
      fprintf(fp, "nextPosChangeFrame = %d\n", nextPosChangeFrame);
      fprintf(fp, "xPos = %d\n", xPos);
      fprintf(fp, "yPos = %d\n", yPos);
      fprintf(fp, "size of vDrops vector = %d\n", size);
      for (int k=0; k<size; k++) {
         fprintf(fp, "   element %d: centerX = %d, centerY = %d, speed = %f, radius = %f, on = %d\n",
                     k, vDrops[k].centerX, vDrops[k].centerY, vDrops[k].speed, vDrops[k].radius, vDrops[k].on);
      }
      fclose(fp);
   }
   return PV_SUCCESS;
}

} // namespace PV
