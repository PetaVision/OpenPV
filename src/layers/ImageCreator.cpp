/*
 * ImageCreator.cpp
 *
 *  Created on: Aug 25, 2009
 *      Author: Shreyas
 */

#include <cmath>
#include <cassert>

#include "../io/tiff.h"
#include "../include/pv_common.h"
#include "../utils/pv_random.h"

#include "ImageCreator.hpp"

namespace PV {
/** Constructor for ImageCreator class.
 * initialize_data() allocates memory for data;
 * data lives in an extended frame of size
 * (nx+2nPad)*(ny+2nPad)*nBands
 * initialize() allocates memory for drawBuffer;
 *  drawBuffer lives in a restricted frame
 * of size nx*ny*nBands
 */
ImageCreator::ImageCreator(const char * name, HyPerCol * hc) : Image(name, hc)
{
   initialize();
#ifdef OBSOLETE
   updateImage(0.0, 0.0);
#endif
}

ImageCreator::~ImageCreator()
{
   free(drawBuffer);
}

/**
 *
 * drawBuffer lives in a restricted frame
 * of size nx*ny*nBands
 */
int ImageCreator::initialize()
{
   const PVLayerLoc * loc = getLayerLoc();
   const int numItems = loc->nx * loc->ny * loc->nBands;

   drawBuffer = (unsigned char *) calloc(sizeof(unsigned char), numItems);
   assert(drawBuffer != 0);

   return 0;
}

/**
 * Description: Updates the image drawBuffer. Can be called for
 *              every iteration from Retina. Desired periodic changes
 *              to the images to be done here.
 *
 * Arguments: time - current time-step.
 *            dt   - difference between timesteps.
 *            targetbuf - the buffer passed (usually from Retina) to which
 *                        the new image must be written.
 *            time and dt can be used to name the images written to files.
 *
 * Return value: 0 if successful, else non-zero.
 *
 * NOTE: 1) drawBuffer lives in an restricted frame (no bundaries)
 *       2) THIS METHOD NEEDS TO BE DESIGNED FOR EACH EXPERIMENT!!!
 *
 */
bool ImageCreator::updateImage(float time_step, float dt)
{
   const PVLayerLoc * loc = getLayerLoc();
   static int prevposx = 0, prevposy = 0;

   static int posx = 5;
   static int posy = 10;
   int lengtha = 10;
   int lengthb = 15;

   clearImage();

   if (time_step > 0.0) { // do not change initial values
      posx += threewaytoss(0.0, 1.0, 0.0); // go back, stay, go ahead
      if (posx < 0) {
         posx = 0;
      } else if ((posx + lengtha) >= loc->nx) {
         posx = (loc->nx - lengtha);
      }

      posy += threewaytoss(0.0, 1.0, 0.0);
      if (posy < 0) {
         posy = 0;
      }
      else if ((posy + lengthb) >= loc->ny) {
         posy = (loc->ny - lengthb);//loc->ny-length?
      }
   }

   if ((prevposx != posx) || (prevposy != posy)) {
      modified = true;
      //printf("%f: new image posx = %d posy = %d\n",time_step, posx,posy);
   }
   else {
      //printf("%f: same image posx = %d posy = %d\n",time_step, posx, posy);
   }

   Point2D newpos(posx, posy);
   //drawSquare(newpos, length, 0);
   drawRectangle(newpos,lengtha,lengthb,0);

   if (modified) {
      writeImageToFile(time_step, TXT | TIF | BIN);
      modified = false;
   }

   prevposx = posx;
   prevposy = posy;

   this->copyFromInteriorBuffer(drawBuffer, 1.0f);

   return true;
}


/**
 * Description: Clears the image buffer.
 *
 * Arguments: None
 *
 * Return value: 0 if successful, else non-zero.
 */
int ImageCreator::clearImage()
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   for (int i = 0; i < (nx * ny); i++) {
      mark(i, 0);
   }
   return 0;
}


/**
 * Description: Creates an image with all its pixels set.
 *
 * Arguments: None
 *
 * Return value: 0 if successful, else non-zero.
 */
int ImageCreator::fillImage(pvdata_t val)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   for (int i = 0; i < (nx * ny); i++) {
      mark(i, val);
   }
   return 0;
}


/*
 * Description: Mark pixels on buf randomly.
 *
 * Arguments: None
 *
 * Return value: 0
 */
int ImageCreator::createRandomImage()
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   assert(drawBuffer != NULL); //ToDo: Validation inadequate.
                        //      Check for buf size > (nx * ny)

   for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         drawBuffer[i + j * nx] = (unsigned char) (pv_random() % 2);
      }
         //Fill in all pixels randomly
   }

   return 0;
}


/*
 * Description: Creates n_images number of images of "random" shapes.
 *
 * Arguments: n_images - the number of images to be drawn.
 *
 * Return value: 0 if successful else non-zero.
 */
int ImageCreator::drawMultipleRandomShapes(int n_images)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   clearImage();
   unsigned int posx; //random() % nx;
   unsigned int posy; //random() % ny;
   unsigned int length = 4; //nx / 8;

   for (int image = 0; image < n_images; image++) {
      do {
         posx = pv_random() % nx;
      } while(posx < 0 && posx > (nx - length));

      do{
         posy = pv_random() % ny;
      } while(posy < 0 || posy > (nx - length));

      Point2D newpos(posx, posy);
      drawSquare(newpos, length, 0);
   }
   return 0;
}


/*
 * Description: Draw a line given an end point and other parameters.
 *              Lines are drawn using Bresenham's algorithm.
 *
 * Arguments: origin: One of the end points of the line segment.
 *            length:
 *            theta: angle formed by the line with the horizontal in degrees.
 *
 * Return value: 0 if successful, non-zero otherwise.
 */
int ImageCreator::drawLine(Point2D origin, unsigned int length, float theta)
{
   const PVLayerLoc * loc = getLayerLoc();

   int x1 = origin.getX();
   int y1 = origin.getY();
   int x2 = approx(x1 + ((length - 1) * cos(deg2rad(theta))));
   int y2 = approx(y1 + ((length - 1) * sin(deg2rad(theta))));
   /** length is decreased by one since the start point, (x1, y1) is also
    *  included in the line.
    */

   if ((x1 < 0) || (y1 < 0) || (x2 >= loc->nx) || (y2 >= loc->ny)) {
      std::cerr << "Error: Cannot draw line from (%d, %d) of length at an angle .\n";
      return 1;
   }
   return (drawBresenhamLine(x1, y1, x2, y2));
}

/*
 * Description: Draw lines between two points.
 *
 * Arguments: The two end points.
 *
 * Return value: None.
 */
int ImageCreator::drawLine(Point2D pt1, Point2D pt2)
{
   const PVLayerLoc * loc = getLayerLoc();

   int x1 = pt1.getX();
   int y1 = pt1.getY();
   int x2 = pt2.getX();
   int y2 = pt2.getY();

   if ((x1 < 0) || (y1 < 0) || (x2 >= loc->nx) || (y2 >= loc->ny)) {
      std::cerr << "Error: Cannot draw line between"; // << pt1 << " and "<< pt2 << ".\n";
      return 1;
   }
   return (drawBresenhamLine(pt1.getX(), pt1.getY(), pt2.getX(), pt2.getY()));
}

/*
 * Description: Draw square.
 *
 * Arguments: origin: left-top vertex of the square.
 *            lengtha: side of the square.
 *            theta: angle that lengtha forms with the horizontal.
 *                   theta must be in degrees.
 */
int ImageCreator::drawSquare(Point2D origin, unsigned int length, unsigned int theta)
{
   return (drawRectangle(origin, length, length, theta));
}

/*
 * Description: Draw square. If the vertices passed do not form a
 *              square, a quadrilateral is drawn if the points are valid.
 *
 * Arguments: Vertices of the square either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 */
int ImageCreator::drawSquare(Point2D pt1, Point2D pt2, Point2D pt3, Point2D pt4)
{
   return (drawQuadrilateral(pt1, pt2, pt3, pt4));
}

/*
 * Description: Draw rectangle.
 *
 * Arguments: origin: left-top vertex of the rectangle.
 *            lengtha, lengthb: two sides of the rectangle.
 *            theta: angle that lengtha forms with the horizontal.
 *                   theta must be in degrees.
 */
int ImageCreator::drawRectangle(Point2D origin, unsigned int lengtha,
                             unsigned int lengthb, unsigned int theta)
{
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;

   int err = 0;
   int x1 = origin.getX();
   int y1 = origin.getY();

   int x2 = approx(x1 + ((lengtha - 1) * cos(deg2rad(theta))));
   int y2 = approx(y1 - ((lengtha - 1) * sin(deg2rad(theta))));

   int x3 = approx(x2 + ((lengthb - 1) * sin(deg2rad(theta))));
   int y3 = approx(y2 + ((lengthb - 1) * cos(deg2rad(theta))));

   int x4 = approx(x1 + ((lengthb - 1) * sin(deg2rad(theta))));
   int y4 = approx(y1 + ((lengthb - 1) * cos(deg2rad(theta))));

   if ((x1 < 0) || (x2 < 0) || (x3 < 0) || (x4 < 0) ||
       (y1 < 0) || (y2 < 0) || (y3 < 0) || (y4 < 0) ||
       (x1 >= nx) || (x2 >= nx) || (x3 >= nx) || (x4 >= nx) ||
       (y1 >= ny) || (y2 >= ny) || (y3 >= ny) || (y4 >= ny)) {

      std::cout << "ImageCreator: Error drawing 2D figure "      // << origin << ".\n";
      << "x1 " << x1 << "y1 " << y1
      << "x2 " << x2 << "y2 " << y2
      << "x3 " << x3 << "y3 " << y3
      << "x4 " << x4 << "y4 " << y4;
      return 1;
   }

   err |= drawBresenhamLine(x1, y1, x2, y2);
   err |= drawBresenhamLine(x2, y2, x3, y3);
   err |= drawBresenhamLine(x3, y3, x4, y4);
   err |= drawBresenhamLine(x4, y4, x1, y1);

   return err;
}

/*
 * Description: Draw rectangle. If the vertices passed do not form a
 *              rectangle, a quadrilateral is drawn if the points are valid.
 *
 * Arguments: Vertices of the rectangle either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 *
 * Return value: None.
 */
int ImageCreator::drawRectangle(Point2D pt1, Point2D pt2,
                             Point2D pt3, Point2D pt4) {

   return (drawQuadrilateral(pt1, pt2, pt3, pt4));
}

/*
 * Description: Draw quadrilateral.
 *
 * Arguments: Vertices of the quadrilateral either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 *
 * Return value: None.
 */
int ImageCreator::drawQuadrilateral(Point2D pt1, Point2D pt2,
                           Point2D pt3, Point2D pt4)
{
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;

   int err = 0;
   int x1 = pt1.getX();
   int y1 = pt1.getY();

   int x2 = pt2.getX();
   int y2 = pt2.getY();

   int x3 = pt3.getX();
   int y3 = pt3.getY();

   int x4 = pt4.getX();
   int y4 = pt4.getY();

   if ((x1 < 0) || (x2 < 0) || (x3 < 0) || (x4 < 0) ||
       (y1 < 0) || (y2 < 0) || (y3 < 0) || (y4 < 0) ||
       (x1 >= nx) || (x2 >= nx) || (x3 >= nx) || (x4 >= nx) ||
       (y1 >= ny) || (y2 >= ny) || (y3 >= ny) || (y4 >= ny)) {

         std::cout << "ImageCreator: Error drawing 2D figure at ";// << pt1 << ".\n";
         return 1;
      }

   err |= drawBresenhamLine(x1, y1, x2, y2);
   err |= drawBresenhamLine(x2, y2, x3, y3);
   err |= drawBresenhamLine(x3, y3, x4, y4);
   err |= drawBresenhamLine(x4, y4, x1, y1);

   return err;
}

/*
 * Description: Draw lines using Bresenham's algorithm.
 *
 * Arguments: (x1, y1): Point 1
 *            (x2, y2): Point 2
 *
 * Return value: None
 *
 * ToDo: Rewrite
 */
int ImageCreator::drawBresenhamLine(int x1, int y1, int x2, int y2) {

   int deltax = x2 - x1;
   int deltay = y2 - y1;
   int steep = (abs(deltay) >= abs(deltax));
   if (steep) {
       swap(x1, y1);
       swap(x2, y2);
       deltax = x2 - x1;
       deltay = y2 - y1;
   }
   int xstep = 1;
   if (deltax < 0) {
       xstep = -1;
       deltax = -deltax;
   }
   int ystep = 1;
   if (deltay < 0) {
       ystep = -1;
       deltay = -deltay;
   }
   int E = (2 * deltay) - deltax;
   int y = y1;
   int xDraw, yDraw;
   for (int x = x1; x != x2; x += xstep) {
       if (steep) {
           xDraw = y;
           yDraw = x;
       } else {
           xDraw = x;
           yDraw = y;
       }
       mark(xDraw, yDraw, 1);
       if (E > 0) {
           E += (2 * deltay) - (2 * deltax);
           y = y + ystep;
       } else {
           E += (2 * deltay);
       }
   }
   return 0;
}

/*
 * Description: Marks pixels on 2D-buf.
 *              Has sharp edges - does not validate inputs (for speed)
 *              Behavior for illegal arguments is undefined.
 *
 * Arguments: i: x coordinate of the point to be plotted.
 *            j: y coordinate of the point to be plotted.
 *            value: the value to be written to the image at (i, j).
 *
 * Return value:
 */
inline void ImageCreator::mark(unsigned int i, unsigned int j, int value)
{
   drawBuffer[i + j * getLayerLoc()->nx] = value;
}

/*
 * Description: Marks pixels on 2D-buf given the index on the linearized buffer
 *              Has sharp edges - does not validate inputs (for speed)
 *              DO NOT use this function unless absolutely sure.
 *              Behavior for illegal arguments is undefined.
 *
 * Arguments: i: index on the imagebuffer whose value need to be set to value.
 *            value: the value to be written to the image at index i.
 *
 * Return value:
 */
inline void ImageCreator::mark(unsigned int i, int value)
{
   drawBuffer[i] = (unsigned char) value;
}

/*
 * Description: Gets pixel values on 2D-buf.
 *              Has sharp edges - does not validate inputs (for speed)
 *              Behavior for illegal arguments is undefined.
 *
 * Arguments: i: x coordinate of the point.
 *            j: y coordinate of the point.
 *
 * Return value: the value at (i, j) in the image.
 */
inline unsigned char ImageCreator::getmark(unsigned int i, unsigned int j)
{
   return drawBuffer[i + j * getLayerLoc()->nx];
}


/*
 * Description: Writes images to files. Provides the option of writing to
 *              bin, txt and tif.
 *
 * Arguments: time - current timestep, used to create file name with the
 *                   timestep on it
 *            options - can be BIN, TXT or TIF.
 *                      Multiple options can be provided using |.
 *                      For eg: BIN | TIF would create both bin and tif files
 *                      with the same filename but different file extensions.
 *
 * Problem: tiff_write_file() needs a float * argument (data in the extended frame)
 * while we want to write unsigned char * drawBuffer, which lives
 * in the restricted frame.
 *
 * Return value: 0 if successful, non-zero otherwise.
 */
int ImageCreator::writeImageToFile(const float time, const unsigned char options)
{
   int status = 0;
   const PVLayerLoc * loc = getLayerLoc();
   unsigned int istxton = 0, istifon = 0, isbinon = 0;

   char basicfilename[128] = { 0 };
   char tiffilename[256] = { 0 }, txtfilename[256] = { 0 }, binfilename[256] = { 0 };

   snprintf(basicfilename, 127, "img2D_%.2f", time);
   istifon = ((options >= 4) ? 1 : 0);
   istxton = (((options == 7) || (options == 3) || (options == 2)) ? 1 : 0);
   isbinon = ((options % 2) ? 1 : 0);

   if (istifon) {
      snprintf(tiffilename, 255, "%simages/%s.tif", OUTPUT_PATH, basicfilename);
      //status |= tiff_write_file(tiffilename, data, loc->nx, loc->ny);
      status |= tiff_write_file_drawBuffer(tiffilename, drawBuffer, loc->nx, loc->ny);
   }
   if (istxton) {
      snprintf(txtfilename, 255, "%simages/%s.txt", OUTPUT_PATH, basicfilename);
      status |= writeImageToTxt(txtfilename);
   }
   if (isbinon) {
      snprintf(binfilename, 255, "%simages/%s.bin", OUTPUT_PATH, basicfilename);
      status |= writeImageToBin(binfilename);
   }

   return status;
}

/**
 * Description: Writes image to .txt file.
 *
 * Arguments: filename - file to be written to.
 *
 * getmark() returns information from drawBuffer which lives
 * in the restricted frame.
 *
 * Return value: 0 if successful, non-zero otherwise.
 */
int ImageCreator::writeImageToTxt(const char *filename)
{
   FILE * txtfile = NULL;
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   if ((txtfile = fopen(filename, "w"))) {
      for (int i = 0; i < nx; i++) {
         for (int j = 0; j < ny; j++) {
            fprintf(txtfile, "%d ", (int) getmark(i, j));
         }
         fprintf(txtfile, "\n");
      }
      fclose(txtfile);
      return 0;
   }
   else {
      fprintf(stderr, "Error: Cannot open .txt for writing image.\n");
      return 1;
   }
}

/**
 * Description: Writes image to .bin file.
 *
 * Arguments: filename - file to be written to.
 *
 * getmark() returns information from drawBuffer which lives
 * in the restricted frame.
 *
 * Return value: 0 if successful, non-zero otherwise.
 */
int ImageCreator::writeImageToBin(const char *filename)
{
   FILE * binfile = NULL;
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;

   if ((binfile = fopen(filename, "wb"))) {
      for (int i = 0; i < nx; i++) {
         for (int j = 0; j < ny; j++) {
            fprintf(binfile, "%d ", (int) getmark(i, j));
         }
         fprintf(binfile, "\n");
      }
      fclose(binfile);
      return 0;
   }
   else {
      fprintf(stderr, "Error: Cannot open .bin for writing image.\n");
      return 1;
   }
}


/*
 * Description: Copy the data from buf maintained by ImageCreator class to
 *              the buffer pass as an argument.
 *
 * Arguments: targetbuf - the buffer to be written to.
 *
 * Return value: 0 if successful, non-zero otherwise.
 */
int ImageCreator::copyImage(pvdata_t * targetbuf)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   for (int i = 0; i < (nx * ny); i++) {
      targetbuf[i] = (pvdata_t) drawBuffer[i];
   }
   return 0;
}

/*
 * Description: Returns -1, 0 or 1 randomly. The probabilities of each can be
 *              tuned with the arguments.
 *
 * Arguments: probBack - The probability of -1 being returned.
 *            probStay - The probability of 0 being returned.
 *            probForward - The probability of 1 being returned.
 *            The three values must be <=1 and their sum must add up to 1.
 *
 * Return value: -1, 0 or 1 with the probabilities passed as arguments.
 *               With illegal aruments, the behavior is undefined.
 */
int ImageCreator::threewaytoss(double probBack, double probStay, double probForward)
{
   int decider = pv_random() % 100;
   float total_prob = decider / 100.00;

   if ((total_prob >= 0) && (total_prob < probStay))
      return 0;
   else if ((total_prob >= probStay) && (total_prob < (probStay + probBack)))
      return -1;
   else
      return 1;
}

/*
 * Description: Utility function for swapping variables.
 *
 * Arguments: References to variables.
 *
 * Return value: None.
 *
 * ToDo: Convert to template if necessary.
 */
void ImageCreator::swap(int &a, int &b)
{
   int temp = a;
   a = b;
   b = temp;
}

/*
 * Description: Utility function to convert degrees to radians.
 *
 * Arguments: angleInDegrees
 *
 * Return value: angle in radians.
 */
inline double ImageCreator::deg2rad(int angleInDegrees)
{
   return (0.0174532925 * (angleInDegrees % 360));
}

/*
 * Description: Approximates a double, numbers >= X.5 are rounded to (X + 1)
 *              numbers < X.5 to X.0. Behaviour undefined for negative numbers.
 *
 * Arguments: temp: Number to be approximated. Must be positive.
 *
 * Return value: if temp is positive, approximate value of temp,
 *               else undefined.
 */
inline int ImageCreator::approx(double temp)
{
   return (int)(temp + 0.5);
}

/*
 * Description: Test function to test the images written to the buffer.
 *              Writes "*" is pixel is turned on, "." otherwise.
 *
 * Arguments: None
 *
 * Return value: None
 */
void ImageCreator::testImage()
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   std::cout << "\n\n";
   for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         if ( drawBuffer[i + j * nx] == 1)
            std::cout << "* ";
         else
            std::cout << ". ";
      }
      std::cout << "\n";
   }
}

}

