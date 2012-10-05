/*
 * Patterns.cpp
 *
 *  Created on: April 21, 2010
 *      Author: Marian Anghel and Craig Rasmussen
 */

#include "Patterns.hpp"
#include <src/include/pv_common.h>  // for PI
#include <src/utils/pv_random.h>
#include "stdio.h"

#define PATTERNS_MAXVAL  1.0f

namespace PV {

// CER-new
FILE * fp;
int start = 0;

Patterns::Patterns() {
   initialize_base();
}

Patterns::Patterns(const char * name, HyPerCol * hc, PatternType type) {
   initialize_base();
   initialize(name, hc, type);
}

int Patterns::initialize_base() {
   patternsOutputPath = NULL;

   return PV_SUCCESS;
}

int Patterns::initialize(const char * name, HyPerCol * hc, PatternType type) {
   Image::initialize(name, hc, NULL);
   this->type = type;

   // CER-new
   fp = fopen("bar-pos.txt", "w");

   this->type = type;

   // set default params
   // set reference position of bars
   this->prefPosition = -3; // does not get it from the params file !
   this->position = this->prefPosition;
   this->lastPosition = this->prefPosition;

   // set bars orientation to default values
   this->orientation = vertical;
   this->lastOrientation = orientation;

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

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);

   //clearPattern(PATTERNS_MAXVAL);

   // make sure initialization is finished
   updateState(0.0, 0.0);
   return PV_SUCCESS;
}

Patterns::~Patterns()
{
   // CER-new
   fclose(fp);
}

int Patterns::tag()
{
   if (orientation == vertical) return position;
   else                         return 10*position;
}

int Patterns::initPattern(float val,float time)
{
   float width, height;


   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int sx = 1;
   const int sy = sx * nx;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = 0.18;               // make it 18% gray
   }

     val = 0.75;                    // box value is 196

   if (type == RECTANGLES) {
      // width  = minWidth  + (maxWidth  - minWidth)  * pv_random_prob();
      // height = minHeight + (maxHeight - minHeight) * pv_random_prob();

	  width = 60;
	  height = 60;

      const int half_w = width/2;
      const int half_h = height/2;

      // random center location
      //const int xc = (nx-1) * pv_random_prob();
      //const int yc = (ny-1) * pv_random_prob();

      int xc = 60+(time-100)/2;
      fprintf(stdout,"This is xc %i and at time %f ",xc,time);
      int yc = 60;

      int x0 = (xc - half_w < 0) ? 0 : xc - half_w;
      int y0 = (yc - half_h < 0) ? 0 : yc - half_h;

      int x1 = (xc + half_w > nx) ? nx : xc + half_w;
      int y1 = (yc + half_h > ny) ? ny : yc + half_h;

      for (int iy = y0; iy < y1; iy++) {
    	 fprintf(stdout,"This is yc %i and iy %i\n",yc,iy);
         for (int ix = x0; ix < x1; ix++) {
            data[ix * sx + iy * sy] = val;
         }
      }

      xc = 60;
      yc = 180;

            x0 = (xc - half_w < 0) ? 0 : xc - half_w;
            y0 = (yc - half_h < 0) ? 0 : yc - half_h;

            x1 = (xc + half_w > nx) ? nx : xc + half_w;
            y1 = (yc + half_h > ny) ? ny : yc + half_h;

            for (int iy = y0; iy < y1; iy++) {
               for (int ix = x0; ix < x1; ix++) {
                  data[ix * sx + iy * sy] = val;
               }
            }
            xc = 180;
             yc =180;

                   x0 = (xc - half_w < 0) ? 0 : xc - half_w/2.;
                   y0 = (yc - half_h < 0) ? 0 : yc - half_h/2.;

                   x1 = (xc + half_w > nx) ? nx : xc + half_w/2.;
                   y1 = (yc + half_h > ny) ? ny : yc + half_h/2.;

                   for (int iy = y0; iy < y1; iy++) {
                      for (int ix = x0; ix < x1; ix++) {
                         data[ix * sx + iy * sy] = val;
                      }
                   }


      position = x0 + y0*nx;
      return 0;
   }

   // type is bars

   int box = 40; // box half width

   if (orientation == vertical) { // vertical bars
      width = maxWidth;
      fprintf(stdout,"VERTICAL \n");
      for (int iy = 0; iy < ny; iy++) {
         for (int ix = 0; ix < nx; ix++) {
        	 if ((ix > nx/2-box) and (ix<nx/2+box) and (iy > ny/2-box) and (iy<ny/2+box) ) { // clear center
        		 data[ix * sx + iy * sy]  = 0;
        		 //fprintf(stdout, "Pattern center pixel is  %i %i %i %i %i %i \n",ix,nx/2-1,nx/2+1,iy,ny/2-1,ny/2+1);
        		 if ((ix > nx/2-1) and (ix<nx/2+1) and (iy > ny/2-1) and (iy<ny/2+1) ) data[ix * sx + iy * sy]  = .5;
        		 if ((ix > nx/2-box/2-2) and (ix<nx/2-box/2+2) and (iy > ny/2+box/2-2) and (iy<ny/2+box/2+2) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2-box/2-3) and (ix<nx/2-box/2+3) and (iy > ny/2-box/2-3) and (iy<ny/2-box/2+3) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2+box/2-4) and (ix<nx/2+box/2+4) and (iy > ny/2+box/2-4) and (iy<ny/2+box/2+4) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2+box/2-5) and (ix<nx/2+box/2+5) and (iy > ny/2-box/2-5) and (iy<ny/2-box/2+5) ) data[ix * sx + iy * sy]  = .8;
      	 }
        	 else
        	 {
        	 int m = (ix + position) % int(2.*width);
             data[ix * sx + iy * sy] = (m < width) ? val*(1.-float(ix+position-m)/floor(2.*width)/5.) : 0;
             //data[ix * sx + iy * sy] = (m < width) ? val : 0;
            // if(m<width){
            //    fprintf(stdout, "Pattern +1 ix factor %i %i %i %f \n",ix+position, m,ix+position-m,float(ix+position-m)/floor(2.*width)/10.);
                //fprintf(stdout, "Pattern +1 ix factor %i %f %i \n",ix+position, width,int(nx/width));
            //     }
            //else{
            //fprintf(stdout, "Pattern ix factor %i %f \n",ix, 0.0 );
            //}

        	 }

             //if (iy == 0) {
             //	fprintf(stdout, "Pattern x vs. content: %i %f \n",ix,data[ix * sx + iy * sy]);
             //}
        	 }

      }
   }
   else { // horizontal bars
      height = maxHeight;
      for (int iy = 0; iy < ny; iy++) {
         int m = (iy + position) % int(2*height);
         for (int ix = 0; ix < nx; ix++) {
        	 if ((ix > nx/2-box) and (ix<nx/2+box) and (iy > ny/2-box) and (iy<ny/2+box) ) { // clear center
        		 data[ix * sx + iy * sy]  = 0;
//        		 fprintf(stdout, "horizontal Pattern center pixel is  %i %i %i %i %i %i \n",ix,nx/2-1,nx/2+1,iy,ny/2-1,ny+1);
        		 if ((ix > nx/2-1) and (ix<nx/2+1) and (iy > ny/2-1) and (iy<ny/2+1) ) {
        					 data[ix * sx + iy * sy]  = 1.;
        		 }
        		 if ((ix > nx/2-box/2-2) and (ix<nx/2-box/2+2) and (iy > ny/2+box/2-2) and (iy<ny/2+box/2+2) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2-box/2-3) and (ix<nx/2-box/2+3) and (iy > ny/2-box/2-3) and (iy<ny/2-box/2+3) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2+box/2-4) and (ix<nx/2+box/2+4) and (iy > ny/2+box/2-4) and (iy<ny/2+box/2+4) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2+box/2-5) and (ix<nx/2+box/2+5) and (iy > ny/2-box/2-5) and (iy<ny/2-box/2+5) ) data[ix * sx + iy * sy]  = 1.;
      	 }
        	 else
        	 {
            data[ix * sx + iy * sy] = (m < height) ? val*(1.-float(iy+position-m)/floor(2.*height)/10.) : 0;
        	 }

      }
   }
   }
   return 0;
}
int Patterns::initPattern(float val)
{
   float width, height;


   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int sx = 1;
   const int sy = sx * nx;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = 0.0;
   }

   if (type == RECTANGLES) {
      // width  = minWidth  + (maxWidth  - minWidth)  * pv_random_prob();
      // height = minHeight + (maxHeight - minHeight) * pv_random_prob();

	  width = 60;
	  height = 60;

      const int half_w = width/2;
      const int half_h = height/2;

      // random center location
      //const int xc = (nx-1) * pv_random_prob();
      //const int yc = (ny-1) * pv_random_prob();

      int xc = 60;
      int yc = 60;

      int x0 = (xc - half_w < 0) ? 0 : xc - half_w;
      int y0 = (yc - half_h < 0) ? 0 : yc - half_h;

      int x1 = (xc + half_w > nx) ? nx : xc + half_w;
      int y1 = (yc + half_h > ny) ? ny : yc + half_h;

      for (int iy = y0; iy < y1; iy++) {
    	 fprintf(stdout,"This is yc %i and iy %i\n",yc,iy);
         for (int ix = x0; ix < x1; ix++) {
            data[ix * sx + iy * sy] = val;
         }
      }

      xc = 60;
      yc = 180;

            x0 = (xc - half_w < 0) ? 0 : xc - half_w;
            y0 = (yc - half_h < 0) ? 0 : yc - half_h;

            x1 = (xc + half_w > nx) ? nx : xc + half_w;
            y1 = (yc + half_h > ny) ? ny : yc + half_h;

            for (int iy = y0; iy < y1; iy++) {
               for (int ix = x0; ix < x1; ix++) {
                  data[ix * sx + iy * sy] = val;
               }
            }


      position = x0 + y0*nx;
      return 0;
   }

   // type is bars

   int box = 40; // box half width

   if (orientation == vertical) { // vertical bars
      width = maxWidth;
      fprintf(stdout,"VERTICAL \n");
      for (int iy = 0; iy < ny; iy++) {
         for (int ix = 0; ix < nx; ix++) {
        	 if ((ix > nx/2-box) and (ix<nx/2+box) and (iy > ny/2-box) and (iy<ny/2+box) ) { // clear center
        		 data[ix * sx + iy * sy]  = 0;
        		 //fprintf(stdout, "Pattern center pixel is  %i %i %i %i %i %i \n",ix,nx/2-1,nx/2+1,iy,ny/2-1,ny/2+1);
        		 if ((ix > nx/2-1) and (ix<nx/2+1) and (iy > ny/2-1) and (iy<ny/2+1) ) data[ix * sx + iy * sy]  = .5;
        		 if ((ix > nx/2-box/2-2) and (ix<nx/2-box/2+2) and (iy > ny/2+box/2-2) and (iy<ny/2+box/2+2) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2-box/2-3) and (ix<nx/2-box/2+3) and (iy > ny/2-box/2-3) and (iy<ny/2-box/2+3) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2+box/2-4) and (ix<nx/2+box/2+4) and (iy > ny/2+box/2-4) and (iy<ny/2+box/2+4) ) data[ix * sx + iy * sy]  = .8;
        		 if ((ix > nx/2+box/2-5) and (ix<nx/2+box/2+5) and (iy > ny/2-box/2-5) and (iy<ny/2-box/2+5) ) data[ix * sx + iy * sy]  = .8;
      	 }
        	 else
        	 {
        	 int m = (ix + position) % int(2.*width);
             data[ix * sx + iy * sy] = (m < width) ? val*(1.-float(ix+position-m)/floor(2.*width)/5.) : 0;
             //data[ix * sx + iy * sy] = (m < width) ? val : 0;
            // if(m<width){
            //    fprintf(stdout, "Pattern +1 ix factor %i %i %i %f \n",ix+position, m,ix+position-m,float(ix+position-m)/floor(2.*width)/10.);
                //fprintf(stdout, "Pattern +1 ix factor %i %f %i \n",ix+position, width,int(nx/width));
            //     }
            //else{
            //fprintf(stdout, "Pattern ix factor %i %f \n",ix, 0.0 );
            //}

        	 }

             //if (iy == 0) {
             //	fprintf(stdout, "Pattern x vs. content: %i %f \n",ix,data[ix * sx + iy * sy]);
             //}
        	 }

      }
   }
   else { // horizontal bars
      height = maxHeight;
      for (int iy = 0; iy < ny; iy++) {
         int m = (iy + position) % int(2*height);
         for (int ix = 0; ix < nx; ix++) {
        	 if ((ix > nx/2-box) and (ix<nx/2+box) and (iy > ny/2-box) and (iy<ny/2+box) ) { // clear center
        		 data[ix * sx + iy * sy]  = 0;
//        		 fprintf(stdout, "horizontal Pattern center pixel is  %i %i %i %i %i %i \n",ix,nx/2-1,nx/2+1,iy,ny/2-1,ny+1);
        		 if ((ix > nx/2-1) and (ix<nx/2+1) and (iy > ny/2-1) and (iy<ny/2+1) ) {
        					 data[ix * sx + iy * sy]  = 1.;
        		 }
        		 if ((ix > nx/2-box/2-2) and (ix<nx/2-box/2+2) and (iy > ny/2+box/2-2) and (iy<ny/2+box/2+2) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2-box/2-3) and (ix<nx/2-box/2+3) and (iy > ny/2-box/2-3) and (iy<ny/2-box/2+3) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2+box/2-4) and (ix<nx/2+box/2+4) and (iy > ny/2+box/2-4) and (iy<ny/2+box/2+4) ) data[ix * sx + iy * sy]  = 1.;
        		 if ((ix > nx/2+box/2-5) and (ix<nx/2+box/2+5) and (iy > ny/2-box/2-5) and (iy<ny/2-box/2+5) ) data[ix * sx + iy * sy]  = 1.;
      	 }
        	 else
        	 {
            data[ix * sx + iy * sy] = (m < height) ? val*(1.-float(iy+position-m)/floor(2.*height)/10.) : 0;
        	 }

      }
   }
   }
   return 0;
}

int Patterns::clearPattern(float val)
{
   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = val;
   }

   return 0;
}

/**
 * update the image buffers
 */
int Patterns::updateState(float time, float dt)
{
   update_timer->start();

   int size = 0;
   int changed = 0;

   //fprintf(stdout,"---- this is updateState of Patterns with time %f and step size %f ----\n",time,dt);

   // alternate between vertical and horizontal bars
   double p = pv_random_prob();

   if (time== 0.0){
	clearPattern(0.0);
    fprintf(stdout,"---- this is updateState of Patterns with CLEAR at %f ----\n",time);

   }

   if(time>49.0 && time<1900)
   {
	           //orientation = vertical;
	           initPattern(PATTERNS_MAXVAL,time);
	           fprintf(stdout,"---- this is updateState of Patterns with INIT %f ----\n",time);
   }


   if (time== 1900.0){
	clearPattern(0.0);
    fprintf(stdout,"---- this is updateState of Patterns with CLEAR %f ----\n",time);
   }




   if (orientation == vertical) { // current vertical gratings
      size = maxWidth;
      if (p < pSwitch) { // switch with probability pSwitch
         orientation = horizontal;
         initPattern(PATTERNS_MAXVAL);
         fprintf(stdout,"horizontal Pattern");
      }
   }
   else {
      size = maxHeight;
      if (p < pSwitch) { // current horizontal gratings
         orientation = vertical;
         initPattern(PATTERNS_MAXVAL);
         fprintf(stdout,"vertical Pattern");
      }
   }

   // moving probability
   double p_move = pv_random_prob();
   if (p_move < pMove) {
      position = calcPosition(position, 2*size);
      //position = (start++) % 4;
      //position = prefPosition;
      initPattern(PATTERNS_MAXVAL);
      fprintf(stdout,"move Pattern");
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
         snprintf(basicfilename, PV_PATH_MAX, "Bars_%.2f.tif", time);
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
int Patterns::calcPosition(int pos, int step)
{
//   float dp = 1.0 / step;        // seems to be not used
   double p = pv_random_prob();
   int random_walk = 1;
   int move_forward = 0;
   int move_backward = 0;
   int random_jump = 0;

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
      }

   return pos;
}

} // namespace PV


