/*
 * Retina2D.hpp
 *
 *  Created on: Jul 9, 2009
 *      Author: Shreyas
 */

#ifndef RETINA2D_HPP_
#define RETINA2D_HPP_

#include <src/layers/Retina.hpp>

namespace PV {

/*
enum Pattern2DType {RANDOM, LINE, SQUARE, RECTANGLE};

class Length2D {

};

class Pattern2D {
public:
   Pattern2D(const Pattern2DType pat);
   Pattern2D(const Pattern2DType pat, const Position2D pos,
             const Length2D len, const float ang);

private:
   Pattern2DType pattern;
   Position2D    position;
   Length2D      length;
   float         angle;
};
*/

class Point2D {
public:
   Point2D(const unsigned int x, const unsigned int y):
             posx(x), posy(y) { };

   unsigned int getX() { return posx; };
   unsigned int getY() { return posy; };

private:
   unsigned int posx, posy;
};

class Retina2D: public PV::Retina {
public:
   //Retina2D();
   Retina2D(const char * name, HyPerCol * hc);
   virtual ~Retina2D();

   int          updateState(float time, float dt);
   virtual int  createImage();
   virtual void clearImage();
   virtual int  createRandomImage();
   //virtual int updateState(float time, float dt);

   void drawLine(Point2D pt1, Point2D pt2);

   void drawLine(Point2D origin,
                 unsigned int length, float theta);

   void drawSquare(Point2D origin,
                   unsigned int length, unsigned int theta);

   void drawSquare(Point2D pt1, Point2D pt2,
                   Point2D pt3, Point2D pt4);

   void drawRectangle(Point2D origin,
                      unsigned int lengtha, unsigned int lengthb,
                      unsigned int theta);

   void drawRectangle(Point2D pt1, Point2D pt2,
                      Point2D pt3, Point2D pt4);

   void drawPolygon(Point2D pt1, Point2D pt2,
                    Point2D pt3, Point2D pt4);

   void testImage();

protected:
   pvdata_t * mom;

private:
   void drawBresenhamLine(int x0, int y0, int x1, int y1);
   void swap(int &a, int &b);
   void plot(unsigned int i, unsigned int j);
   inline double deg2rad(int angleInDegrees);
   inline int    approx(double n);
   int  threewaytoss();
};

}

#endif /* RETINA2D_HPP_ */
