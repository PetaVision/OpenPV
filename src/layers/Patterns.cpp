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
FILE * fp;
int start = 0;

Patterns::Patterns(const char * name, HyPerCol * hc, PatternType type) :
   Image(name, hc)
{
   initializePatterns(name, hc, type);
}

int Patterns::initializePatterns(const char * name, HyPerCol * hc, PatternType type)
{
   // CER-new

   patternsOutputPath = NULL;
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
      this->lastOrientation = orientation;
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
         this->lastOrientation = orientationMode;
      }
      else { //if the set orientation isn't recognized, use vertical as default
         this->orientation = vertical;
         this->lastOrientation = orientation;
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

   minWidth  = 4.0;
   minHeight = 4.0;

   maxWidth  = params->value(name, "width", loc->nx);
   maxHeight = params->value(name, "height", loc->ny);

   pMove   = params->value(name, "pMove", 0.0);
   pSwitch = params->value(name, "pSwitch", 0.0);

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
         printf("Movie output path is not specified in params file.\n"
               "Movie output path set to default \"%s\"\n",patternsOutputPath);
      }
   }

   writePosition     = (int) params->value(name,"writePosition", 0);
   if(writePosition){
      char file_name[PV_PATH_MAX];
      if (patternsOutputPath != NULL){
         //I don't know why someone was saving the return value, but it was
         //generating a compiler warning because it was unused.  Did someone
         //want to use this for something?
         snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", patternsOutputPath);
         //int nchars = snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", patternsOutputPath);
      }
      else{
         snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", hc->getOutputPath());
         //int nchars = snprintf(file_name, PV_PATH_MAX-1, "%s/bar-pos.txt", hc->getOutputPath());
      }
      printf("write position to %s\n",file_name);
      fp = fopen(file_name,"a");
      assert(fp != NULL);
   }

   initPattern(PATTERNS_MAXVAL);

   // make sure initialization is finished
   updateState(0.0, 0.0);

   return EXIT_SUCCESS;
}

Patterns::~Patterns()
{
   // CER-new
   fclose(fp);
}

int Patterns::tag()
{
   if (orientation == vertical)
      return position;
   else
      return 10*position;
}

int Patterns::initPattern(float val)
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


      if (orientation == vertical) { // vertical bars
         width = maxWidth;
         for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
               //int m = (ix + int(position)) % (2*width);
               float m = ix + position - (floor((ix + position) / (2*width))) * (2*width); //calculate position including fraction
               //float m = (ix + position);
               if((int(m)==width)||(int(m)==0)) data[ix * sx + iy * sy] = (m-int(m))*val;
               else if(m<width) data[ix * sx + iy * sy] = val;
               //else if(int(m)==width) data[ix * sx + iy * sy] = (1-(m-int(m)))*val;
               //else if(int(m)==width+1) data[ix * sx + iy * sy] = (m-int(m))*val;
               else data[ix * sx + iy * sy] = 0;
               //data[ix * sx + iy * sy] = (m < width) ? val : 0;
            }
         }
      }
      else { // horizontal bars
         height = maxHeight;
         for (int iy = 0; iy < ny; iy++) {
            //int m = (iy + position) % (2*height);
            float m = iy + position - (floor((iy + position) / (2*height))) * (2*height); //calculate position including fraction
            for (int ix = 0; ix < nx; ix++) {
               if((int(m)==height)||(int(m)==0)) data[ix * sx + iy * sy] = (m-int(m))*val;
               else if(m<height) data[ix * sx + iy * sy] = val;
               //else if(int(m)==height+1) data[ix * sx + iy * sy] = (1-(m-int(m)))*val;
               else data[ix * sx + iy * sy] = 0;
               //data[ix * sx + iy * sy] = (m < height) ? val : 0;
            }
         }
      }

      return 0;
   }

   else if (type == SINEWAVE) {
      if (orientation == vertical) { // vertical bars
         width = maxWidth;
         for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
               int glx = ix+kx0-nb;
               float m = glx + position; //calculate position including fraction

               //sin of 2*pi*m/wavelength, where wavelength=2*width:
               data[ix * sx + iy * sy] = sinf(PI*m/width);
            }
         }
      }
      else { // horizontal bars
         height = maxHeight;
         for (int iy = 0; iy < ny; iy++) {
            int gly = iy+ky0-nb;
            float m = gly + position; //calculate position including fraction
            for (int ix = 0; ix < nx; ix++) {
               //float value=sinf(2*PI*m/height);
               data[ix * sx + iy * sy] = sinf(PI*m/height);
            }
         }
      }
      return 0;
   }

   return 0;
}

/**
 * update the image buffers
 */
int Patterns::updateState(float time, float dt) {
   update_timer->start();

   int size = 0;
   int changed = 0;

   // alternate between vertical and horizontal bars
   double p = pv_random_prob();

   if (orientation == vertical) { // current vertical gratings
      size = maxWidth;
      if (p < pSwitch) { // switch with probability pSwitch
         orientation = horizontal;
         initPattern(PATTERNS_MAXVAL);
      }
   }
   else {
      size = maxHeight;
      if (p < pSwitch) { // current horizontal gratings
         orientation = vertical;
         initPattern(PATTERNS_MAXVAL);
      }
   }

   // moving probability
   double p_move = pv_random_prob();
   if (p_move < pMove) {
      position = calcPosition(position, 2*size);
      //position = (start++) % 4;
      //position = prefPosition;
      initPattern(PATTERNS_MAXVAL);
      //fprintf(fp, "%d %d %d\n", 2*(int)time, position, lastPosition);
   }
   else {
      position = lastPosition;
   }

   if (lastPosition != position || lastOrientation != orientation) {
      lastPosition = position;
      lastOrientation = orientation;
      lastUpdateTime = time;
      changed = 1;
      if (writeImages) {
         char basicfilename[PV_PATH_MAX+1]; // is +1 needed?
         if (type == BARS)
            snprintf(basicfilename, PV_PATH_MAX, "%s/Bars_%.2f.tif", patternsOutputPath, time);
         else if (type == RECTANGLES){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Rectangles_%.2f.tif", patternsOutputPath, time);
         }
         else if (type == SINEWAVE){
            snprintf(basicfilename, PV_PATH_MAX, "%s/Sinewave%.2f.tif", patternsOutputPath, time);
         }
         write(basicfilename);
      }
   }

   update_timer->stop();

   return changed;
}

/**
 *
 *  Return an integer between 0 and (step-1)
 */
float Patterns::calcPosition(float pos, int step)
{
   // float dp = 1.0 / step;
   double p = pv_random_prob();
   /*
    * now use movementType to determine which kind of movement to make
   int random_walk = 1;
   int move_forward = 0;
   int move_backward = 0;
   int random_jump = 0;*/

   /* old code:
   if (random_walk) {
      if (p < 0.5){
         pos = (pos+1) % step;
      } else {
         pos = (pos-1+step) % step;
      }
      //printf("pos = %f\n",position);
   } else if (move_forward){
      pos = (pos+1) % step;
   } else if (move_backward){
      pos = (pos-1+step) % step;
   }
   else if (random_jump) {
      pos = int(p * step) % step;
   }*/

   switch (movementType) {
   case RANDOMWALK:
      if (p < 0.5){
         //pos = (int(pos)+1) % step;
         pos = ((int)(pos+movementSpeed)) % step;
      } else {
         //pos = (int(pos)-1+step) % step;
         pos = ((int)(pos-movementSpeed)) % step;
      }
      break;
   case MOVEFORWARD:
     pos = (pos+movementSpeed) ;
     if(pos>step) {pos -= step;}
     break;
   case MOVEBACKWARD:
      pos = (pos-movementSpeed) ;
      if(pos<0) {pos += step;}
     break;
   case RANDOMJUMP:
      pos = int(p * step) % step;
      break;
   default: //in case of any problems with setting the movementType var, just use the
      //random walk as default
      if (p < 0.5){
         //pos = (int(pos)+1) % step;
         pos = ((int)(pos+movementSpeed)) % step;
      } else {
         //pos = (int(pos)-1+step) % step;
         pos = ((int)(pos-movementSpeed)) % step;
      }
      break;
   }

   return pos;
}

} // namespace PV
