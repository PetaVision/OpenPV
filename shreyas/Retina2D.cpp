/*
 * Retina2D.cpp
 *
 *  Created on: Jul 9, 2009
 *      Author: Shreyas
 */

#include "Retina2D.hpp"
#include <stdlib.h>
#include <assert.h>
#include <iostream>

using namespace std;


namespace PV {
/*
 * Description: Constructor for Retina2D class
 *
 * Arguments:
 *
 * Return value:
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
Retina2D::Retina2D(const char * name, HyPerCol * hc) :
   Retina(name, hc) {
   mom = (pvdata_t *) malloc(clayer->numNeurons * sizeof(pvdata_t));;
//   createImage(clayer->V);
//   createRandomImage(clayer->V);
   cout << "mom: ";
   for (int i = 0; i < clayer->numNeurons; i++) {
     mom[i] = clayer->V[i];
     if (!(i % 2)){
        cout << mom[i];
     }
   }
   cout << "\n";
}


/*
 * Description: Destructor for Retina2D class
 *
 * Arguments: None
 *
 * Return value: None
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */

Retina2D::~Retina2D() {
   free(mom);
}


/*
 * Description:
 *
 * Arguments:
 *
 * Return value:
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
int Retina2D::createImage() {

   pvdata_t * buf = clayer->V;
   const int nx   = clayer->loc.nx;
   const int ny   = clayer->loc.ny;
   const int nf   = clayer->numFeatures;

   static int t = -1;

   int min = 4; // 16;
   int max = nx - min;

   // slide image left and right by one pixel
   t += 1;
   min += t % 2;
   max += t % 2;

   assert(this->clayer->numFeatures == 2);

   srand(time(NULL));

   int posx = 8; //rand() % nx;
   int posy = 8; //rand() % ny;
   unsigned int length = 8; //nx / 8;

   unsigned int deltatime = 1000; //in microsecond for usleep

   Point2D start(posx, posy);
   drawSquare(start, length, 0);

#if 0
   usleep(deltatime);
   clearImage();


   for (;;) {
	   posx += threewaytoss();
	   if (posx < 0) posx = posx;
	   else if (posx >= nx) posx = (nx - 1);

	   posy += threewaytoss();
	   if (posy < 0) posy = posy;
	   else if (posy >= ny) posy = (ny - 1);

	   Point2D newpos(posx, posy);

	   drawSquare(newpos, length, 0);
	   usleep(deltatime);
	   clearImage();
   }
#endif


#if 0
   for (int k = 0; k < clayer->numNeurons; k += 2) {
      int kx = kxPos(k, nx, ny, nf);

      int pat = 1;
      if (kx < min || kx > max) {
         pat = 0;
      }
      buf[k]   = 1 - pat;
      buf[k+1] = pat;
   }
#endif

   return 0;
}

int Retina2D::threewaytoss() {
	int decider = rand() % 3;
	switch (decider) {
	case 0:
		return 0;
	case 1:
		return 1;
	case 2:
		return -1;
	}
}
/*
 * Description: Clears the image buffer.
 *
 * Arguments: None
 *
 * Return value: None
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::clearImage() {
	pvdata_t * buf = clayer->V;

	const int nx   = clayer->loc.nx;
	const int ny   = clayer->loc.ny;

	for (int i = 0; i < (nx * ny); i++) {
		*(buf + i) = 0;
	}
}

/*
 * Description: Draw a line given an end point and other parameters.
 *              Lines are drawn using Bresenham's algorithm.
 *
 * Arguments: origin: One of the end points of the line segment.
 *            length:
 *            theta: angle formed by the line with the horizontal in degrees.
 *
 * Return value:
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawLine(Point2D origin, unsigned int length, float theta) {

   int x1         = origin.getX();
   int y1         = origin.getY();
   int x2         = approx(x1 + (length * cos(deg2rad(theta))));
   int y2         = approx(y1 + (length * sin(deg2rad(theta))));

   drawBresenhamLine(x1, y1, x2, y2);
}

/*
 * Description: Draw lines between two points.
 *
 * Arguments: The two end points.
 *
 * Return value: None.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawLine(Point2D pt1, Point2D pt2) {

   drawBresenhamLine(pt1.getX(), pt1.getY(), pt2.getX(), pt2.getY());
}

/*
 * Description: Draw square.
 *
 * Arguments: origin: left-top vertex of the square.
 *            lengtha: side of the square.
 *            theta: angle that lengtha forms with the horizontal.
 *                   theta must be in degrees.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawSquare(Point2D origin, unsigned int length,
                          unsigned int theta) {

   drawRectangle(origin, length, length, theta);

#if 0
   int x1 = origin.getX();
   int y1 = origin.getY();

   int x2 = approx(x1 + (length * cos(deg2rad(theta))));
   int y2 = approx(y1 - (length * sin(deg2rad(theta))));

   int x3 = approx(x1 + (sqrt(2) * length * cos(deg2rad(45 - theta))));
   int y3 = approx(y1 + (sqrt(2) * length * sin(deg2rad(45 - theta))));

   int x4 = approx(x1 + (length * sin(deg2rad(theta))));
   int y4 = approx(y1 + (length * cos(deg2rad(theta))));

   if ((x1 < 0) || (x2 < 0) || (x3 < 0) || (x4 < 0) ||
       (y1 < 0) || (y2 < 0) || (y3 < 0) || (y4 < 0)) {

      cout << "Retina2D: Error drawing square\n";
      return;
   }

   drawBresenhamLine(x1, y1, x2, y2);
   drawBresenhamLine(x2, y2, x3, y3);
   drawBresenhamLine(x3, y3, x4, y4);
   drawBresenhamLine(x4, y4, x1, y1);
#endif

}

/*
 * Description: Draw square. If the vertices passed do not form a
 *              square, a polygon is drawn if the points are valid.
 *
 * Arguments: Vertices of the square either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawSquare(Point2D pt1, Point2D pt2,
                          Point2D pt3, Point2D pt4) {

   //TODO: add validation
   drawPolygon(pt1, pt2, pt3, pt4);
}

/*
 * Description: Draw rectangle.
 *
 * Arguments: origin: left-top vertex of the rectangle.
 *            lengtha, lengthb: two sides of the rectangle.
 *            theta: angle that lengtha forms with the horizontal.
 *                   theta must be in degrees.
 *
 * Return value: None.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawRectangle(Point2D origin, unsigned int lengtha,
                             unsigned int lengthb, unsigned int theta) {

   int x1 = origin.getX();
   int y1 = origin.getY();

   int x2 = approx(x1 + (lengtha * cos(deg2rad(theta))));
   int y2 = approx(y1 - (lengtha * sin(deg2rad(theta))));

   int x3 = approx(x2 + (lengthb * sin(deg2rad(theta))));
   int y3 = approx(y2 + (lengthb * cos(deg2rad(theta))));

   int x4 = approx(x1 + (lengthb * sin(deg2rad(theta))));
   int y4 = approx(y1 + (lengthb * cos(deg2rad(theta))));

   if ((x1 < 0) || (x2 < 0) || (x3 < 0) || (x4 < 0) ||
       (y1 < 0) || (y2 < 0) || (y3 < 0) || (y4 < 0)) {

      cout << "Retina2D: Error drawing 2D figure.\n";
      return;
   }

   drawBresenhamLine(x1, y1, x2, y2);
   drawBresenhamLine(x2, y2, x3, y3);
   drawBresenhamLine(x3, y3, x4, y4);
   drawBresenhamLine(x4, y4, x1, y1);
}

/*
 * Description: Draw rectangle. If the vertices passed do not form a
 *              rectangle, a polygon is drawn if the points are valid.
 *
 * Arguments: Vertices of the rectangle either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 *
 * Return value: None.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawRectangle(Point2D pt1, Point2D pt2,
                             Point2D pt3, Point2D pt4) {

   drawPolygon(pt1, pt2, pt3, pt4);
}

/*
 * Description: Draw Polygon.
 *
 * Arguments: Vertices of the polygon either in clockwise or
 *            anti-clockwise order. Else behaviour undefined.
 *
 * Return value: None.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawPolygon(Point2D pt1, Point2D pt2,
                           Point2D pt3, Point2D pt4) {

   int x1 = pt1.getX();
   int y1 = pt1.getY();

   int x2 = pt2.getX();
   int y2 = pt2.getY();

   int x3 = pt3.getX();
   int y3 = pt3.getY();

   int x4 = pt4.getX();
   int y4 = pt4.getY();

   if ((x1 < 0) || (x2 < 0) || (x3 < 0) || (x4 < 0) ||
       (y1 < 0) || (y2 < 0) || (y3 < 0) || (y4 < 0)) {

         cout << "Retina2D: Error drawing 2D figure.\n";
         return;
      }

   drawBresenhamLine(x1, y1, x2, y2);
   drawBresenhamLine(x2, y2, x3, y3);
   drawBresenhamLine(x3, y3, x4, y4);
   drawBresenhamLine(x4, y4, x1, y1);
}

/*
 * Description: Draw lines using Bresenham's algorithm.
 *
 * Arguments: (x1, y1): Point 1
 *            (x2, y2): Point 2
 *
 * Return value: None
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::drawBresenhamLine(int x1, int y1, int x2, int y2) {

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
       plot(xDraw, yDraw);
       if (E > 0) {
           E += (2 * deltay) - (2 * deltax);
           y = y + ystep;
       } else {
           E += (2 * deltay);
       }
   }
}

/*
 * Description: Swap variables.
 *
 * Arguments: References to variables.
 *
 * Return value: None.
 *
 * ToDo: Convert to template if necessary.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::swap(int &a, int &b) {

   int temp = a;
   a = b;
   b = temp;
}

/*
 * Description: Marks pixels on 2D-buf.
 *
 * Arguments: i: x coordinate of the point to be plotted.
 *            j: y coordinate of the point to be plotted.
 *
 * Return value:
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::plot(unsigned int i, unsigned int j) {

   pvdata_t * buf = clayer->V;
   const unsigned int nx = clayer->loc.nx;

   *(buf + (i + (j * nx))) = 1;
}

/*
 * Description: Mark pixels on buf randomly.
 *
 * Arguments: None
 *
 * Return value: 0
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
int Retina2D::createRandomImage() {

   pvdata_t * buf = clayer->V;
   assert(buf != NULL); //ToDo: Validation inadequate.
                        //      Check for buf size > (nx * ny)
   assert(this->clayer->numFeatures == 2);

   const unsigned int nx = clayer->loc.nx;
   const unsigned int ny = clayer->loc.ny;

   for (unsigned int i = 0; i < nx; i++) {
      for (unsigned int j = 0; j < ny; j++) {
         *(buf + (i + (j * nx))) = (rand() % 2);
      }
         //Fill in all pixels randomly
   }

   return 0;
}

/*
 * Description: Convert degrees to radians.
 *
 * Arguments: angleInDegrees
 *
 * Return value: angle in radians.
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
inline double Retina2D::deg2rad(int angleInDegrees) {
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
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
inline int Retina2D::approx(double temp) {
	return (((temp - (int) temp) < 0.50) ? ((int) temp) : (((int) temp) + 1));
}

/*
 * Description: Function to test the images written to the buffer.
 *              Writes "*" is pixel is turned on, "." otherwise.
 *
 * Arguments: None
 *
 * Return value: None
 *
 * Change history:
 * 09-June-2009          Shreyas              Function created
 *
 */
void Retina2D::testImage() {

   pvdata_t * buf = clayer->V;
   const unsigned int nx = clayer->loc.nx;
   const unsigned int ny = clayer->loc.ny;

   cout << "\n\n";
   for (unsigned int i = 0; i < nx; i++) {
      for (unsigned int j = 0; j < ny; j++) {
	     if ( *(buf + (i + (j * nx))) == 1)
	    	 cout << "* ";
	     else
	    	 cout << ". ";
	  }
      cout << "\n";
   }
}

#if 0
//Returns the element in buf indexed by i, j
//TODO: Passing nx, ny as parameters can be avoided
//      by designing pvdata_t as a class
inline *pvdata_t Retina2D::IndexedElement
   (pvdata_t *buf, unsigned int nx, unsigned int ny,
      unsigned int i, unsigned int j) {
   return (buf + (i + (j * nx)));
}
#endif

int Retina2D::updateState(float time, float dt)
{
   if (clayer->updateFunc != NULL) {
      clayer->updateFunc(clayer);
   }
   else {
      fileread_params * params = (fileread_params *) clayer->params;
      pvdata_t * activity = clayer->activity->data;
      float* V = clayer->V;

      float poissonEdgeProb  = RAND_MAX * params->poissonEdgeProb;
      float poissonBlankProb = RAND_MAX * params->poissonBlankProb;

      int burstStatus = 0;
      if (params->burstDuration <= 0 || params->burstFreq == 0) {
         burstStatus = sin( 2 * PI * time * params->burstFreq / 1000. ) > 0.;
      }
      else {
         burstStatus = fmod(time/dt, 1000. / (dt * params->burstFreq));
         burstStatus = burstStatus <= params->burstDuration;
      }
      int stimStatus = (time >= params->beginStim) && (time < params->endStim);
      stimStatus = stimStatus && burstStatus;

      if (params->spikingFlag == 0.0) {
         // non-spiking code
         if (stimStatus) {
            for (int k = 0; k < clayer->numNeurons; k++) {
               activity[k] = V[k];
            }
         }
         else {
            for (int k = 0; k < clayer->numNeurons; k++) {
               activity[k] = 0.0;
            }
         }
      }
      else {
         // Poisson spiking...
         const int nf = clayer->numFeatures;
         const int numNeurons = clayer->numNeurons;

         assert(nf > 0 && nf < 3);

         if (stimStatus == 0) {
            // fire at the background rate
            for (int k = 0; k < numNeurons; k++) {
               activity[k] = rand() < poissonBlankProb;
            }
         }
         else {
            // ON case (k even)
            for (int k = 0; k < clayer->numNeurons; k += nf) {
               if ( V[k] == 0.0 )
                  // fire at the background rate
                  activity[k] = (rand() < poissonBlankProb );
               else if ( V[k] > 0.0 )
                  // for gray scale use poissonEdgeProb * abs( V[k] )
                  activity[k] = (rand() < poissonEdgeProb );
               else // V[k] < 0.0
                  // fire at the below background rate (treated as zero if P < 0)
                  activity[k] = (rand() < ( 2 * poissonBlankProb - poissonEdgeProb ) );
            }
            // OFF case (k is odd)
            if (nf == 2) {
               for (int k = 1; k < clayer->numNeurons; k += nf) {
                   if ( V[k] == 0.0 )
                      // fire at the background rate
                      activity[k] = (rand() < poissonBlankProb );
                   else if ( V[k] < 0.0 )
                      // for gray scale use poissonEdgeProb * abs( V[k] )
                      activity[k] = (rand() < poissonEdgeProb );
                   else // V[k] > 0.0
                      // fire at the below background rate (treated as zero if P < 0)
                      activity[k] = (rand() < ( 2 * poissonBlankProb - poissonEdgeProb ) );
                }
            } // nf == 2
         } // stimStatus
      }
   }

   return 0;
}

}
