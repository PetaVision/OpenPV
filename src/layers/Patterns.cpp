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

namespace PV {

// CER-new

Patterns::Patterns() {
   initialize_base();
}

Patterns::Patterns(const char * name, HyPerCol * hc, PatternType type) {
   initialize_base();
   initialize(name, hc, type);
}

int Patterns::initialize_base() {
   patternsOutputPath = NULL;
   patternsFile = NULL;

   return PV_SUCCESS;
}

int Patterns::initialize(const char * name, HyPerCol * hc, PatternType type) {
   Image::initialize(name, hc, NULL);
   assert(this->clayer->loc.nf == 1);
   this->type = type;

   // set default params
   // set reference position of bars
   this->prefPosition = 0; // 3; why was the old default 3???
   this->position = this->prefPosition;
   this->lastPosition = this->prefPosition;


   // set orientation mode
   const char * allowedOrientationModes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "HORIZONTAL",
         "VERTICAL",
         "MIXED",
         "_End_allowedOrientationTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   //if the orientation isn't set, use vertical as the default...
   const char * orientationModeStr = hc->parameters()->stringValue(name, "orientation");
   if( ! orientationModeStr ) {
      this->orientation = vertical;
   }
   else {
      OrientationMode orientationMode;
      int orientationModeMatch = false;
      for( int i=0; strcmp(allowedOrientationModes[i],"_End_allowedOrientationTypes"); i++ ) {
         const char * thisorientationmode = allowedOrientationModes[i];
         if( !strcmp(orientationModeStr, thisorientationmode) ) {
            orientationMode = (OrientationMode) i;
            orientationModeMatch = true;
            break;
         }
      }
      if( orientationModeMatch ) {
         this->orientation = orientationMode;
      }
      else { //if the set orientation isn't recognized, use vertical as default
         this->orientation = vertical;
      }
   }

   //set movement type (random walk is default)
   const char * allowedMovementTypes[] = { // these strings should correspond to the types in enum PatternType in Patterns.hpp
         "RANDOMWALK",
         "MOVEFORWARD",
         "MOVEBACKWARD",
         "RANDOMJUMP",
         "_End_allowedPatternTypes"  // Keep this string; it allows the string matching loop to know when to stop.
   };
   //if the movement type isn't set, use random walk as the default...
   const char * movementTypeStr = hc->parameters()->stringValue(name, "movementType");
   if( ! movementTypeStr ) {
      this->movementType = RANDOMWALK;
   }
   else {
      MovementType movementType;
      int movementTypeMatch = false;
      for( int i=0; strcmp(allowedMovementTypes[i],"_End_allowedPatternTypes"); i++ ) {
         const char * thisMovementType = allowedMovementTypes[i];
         if( !strcmp(movementTypeStr, thisMovementType) ) {
            movementType = (MovementType) i;
            movementTypeMatch = true;
            break;
         }
      }
      if( movementTypeMatch ) {
         this->movementType = movementType;
      }
      else { //if the set movement type isn't recognized, use random walk as default
         this->movementType = RANDOMWALK;
      }
   }

   const PVLayerLoc * loc = getLayerLoc();

   // check for explicit parameters in params.stdp
   //
   PVParams * params = hc->parameters();

   if( type == RECTANGLES ) {
      maxWidth  = params->value(name, "maxWidth", loc->nx);
      maxHeight = params->value(name, "maxHeight", loc->ny);
      minWidth = params->value(name, "minWidth", maxWidth);
      minHeight = params->value(name, "minWeight", maxHeight);
   }
   else {
      maxWidth  = params->value(name, "width", loc->nx); // width of bar when bar is vertical
      maxHeight = params->value(name, "height", loc->ny); // height of bar when bar is horizontal
   }

   if(( type == BARS )||(type == COSWAVE)||(type == SINEWAVE)||
         (type == COSV)||(type == SINEV)) {
      wavelengthVert = params->value(name, "wavelengthVert", 2*maxWidth);
      wavelengthHoriz = params->value(name, "wavelengthHoriz", 2*maxHeight);
   }

   pMove   = params->value(name, "pMove", 0.0);
   pSwitch = params->value(name, "pSwitch", 0.0);

   if((type == COSWAVE)||(type == SINEWAVE)||
         (type == COSV)||(type == SINEV))
      rotation = params->value(name, "rotation", 0.0);

   movementSpeed = params->value(name, "movementSpeed", 1); //1 is the old default...

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);
   // set output path for movie frames
   if(writeImages){
      if ( params->stringPresent(name, "patternsOutputPath") ) {
         patternsOutputPath = strdup(params->stringValue(name, "patternsOutputPath"));
         assert(patternsOutputPath != NULL);
      }
      else {
         patternsOutputPath = strdup( hc->getOutputPath());
         assert(patternsOutputPath != NULL);
         printf("patternsOutputPath is not specified in params file.\n"
               "Patterns output path set to default \"%s\"\n",patternsOutputPath);
      }
   }
   initPatternCntr=0;
   writePosition = (int) params->value(name,"writePosition", 0);
   if(writePosition){
      char file_name[PV_PATH_MAX];

      //Return value of snprintf commented out because it was generating an
      //unused-variable compiler warning.
      //
      snprintf(file_name, PV_PATH_MAX-1, "%s/patterns-pos.txt", patternsOutputPath);
      //int nchars = snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", patternsOutputPath);
      printf("write position to %s\n",file_name);
      // TODO In MPI, fp should only be opened and written to by root process
      patternsFile = fopen(file_name,"a");
      assert(patternsFile != NULL);
   }

   maxVal = params->value(name,"maxValue", PATTERNS_MAXVAL);

   displayPeriod = params->value(name,"displayPeriod", 0.0f);
   // displayPeriod = 0 means nextDisplayTime will always >= starting time and therefore the pattern will update every timestep
   nextDisplayTime = hc->simulationTime() + displayPeriod;

   setOrientation(orientation); // Sets positionBound based on orientation

   generatePattern(maxVal);

   return PV_SUCCESS;
}

Patterns::~Patterns()
{
   free(patternsOutputPath);

   if( patternsFile != NULL ) {
      fclose(patternsFile);
      patternsFile = NULL;
   }
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

int Patterns::tag()
{
   if (orientation == vertical)
      return position;
   else
      return 10*position;
}

int Patterns::generatePattern(float val)
{
   int width, height;

   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int nb = loc->nb;
   const int sx = 1;
   const int sy = sx * nx;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;

   const int nxgl = loc->nxGlobal;
   const int nygl = loc->nyGlobal;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = 0.0;
   }

   if (type == RECTANGLES) {
      width  = minWidth  + (maxWidth  - minWidth)  * pv_random_prob();
      height = minHeight + (maxHeight - minHeight) * pv_random_prob();

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
      return 0;
   }
   else if (type == BARS) { // type is bars
      return generateBars(orientation, data, nx, ny, val);
   }
//   else if (type == COSWAVE) {
//      if (orientation == vertical) { // vertical bars
//         width = maxWidth;
//         for (int iy = 0; iy < ny; iy++) {
//            for (int ix = 0; ix < nx; ix++) {
//               int glx = ix+kx0-nb;
//               int gly = iy+ky0-nb;
//               float m = glx*cos(rotation) - gly*sin(rotation)  + position; //calculate position including fraction
//
//               //sin of 2*pi*m/wavelength, where wavelength=2*width:
//               data[ix * sx + iy * sy] = cosf(PI*m/width);
//            }
//         }
//      }
//      else { // horizontal bars
//         height = maxHeight;
//         for (int iy = 0; iy < ny; iy++) {
//            int gly = iy+ky0-nb;
//            for (int ix = 0; ix < nx; ix++) {
//               int glx = ix+kx0-nb;
//               float m = gly*cos(rotation) + glx*sin(rotation)  + position; //calculate position including fraction
//               //float value=sinf(2*PI*m/height);
//               data[ix * sx + iy * sy] = cosf(PI*m/height);
//            }
//         }
//      }
//      return 0;
//   }
   else if((type == COSWAVE)||(type == SINEWAVE)||
           (type == COSV)||(type == SINEV)) {
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
               data[ix * sx + iy * sy] = sin(PI*m/float(wavelength));
            else if((type == COSWAVE)||(type == COSV))
               data[ix * sx + iy * sy] = cos(PI*m/float(wavelength));

         }
      }
      return 0;
   }
   else if (type == IMPULSE) {
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

   return 0;
}

int Patterns::generateBars(OrientationMode ormode, pvdata_t * buf, int nx, int ny, float val) {
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

/**
 * update the image buffers
 */
int Patterns::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
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

   if( p < pSwitch) { // switch with probability pSwitch
      setOrientation(orientation == vertical ? horizontal : vertical);
      newPattern = true;
   }

   // moving probability
   p -= pSwitch; // Doesn't make sense to both switch and move
   if (p >= 0 && p < pMove) {
      newPattern = true;
      //fprintf(fp, "%d %d %d\n", 2*(int)time, position, lastPosition);
   }

   if (newPattern) {
      lastUpdateTime = timef;
      position = calcPosition(position, positionBound);
      generatePattern(maxVal);
      if (writeImages) {
         char basicfilename[PV_PATH_MAX+1]; // is +1 needed?
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
         fclose(fp);
      }
      else {
         fprintf(stderr, "Unable to read from \"%s\"\n", filename);
      }
   }
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
      if( fp != NULL ) {
         status = fwrite(&orientation, sizeof(OrientationMode), 1, fp) == 1 ? status : PV_FAILURE;
         status = fwrite(&position, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&nextDisplayTime, sizeof(float), 1, fp) ? status : PV_FAILURE;
         status = fwrite(&initPatternCntr, sizeof(int), 1, fp) ? status : PV_FAILURE;
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
   }
   return PV_SUCCESS;
}

} // namespace PV
