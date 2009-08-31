/*
 * Image.hpp
 *
 *  Created on: Aug 25, 2009
 *      Author: Shreyas
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "../include/pv_common.h"
#include "../include/pv_types.h"

#include <iostream>

namespace PV {

const unsigned int BIN = 1;
const unsigned int TXT = 2;
const unsigned int TIF = 4;

class Point2D {
public:
   Point2D(const unsigned int x, const unsigned int y):
             posx(x), posy(y) { };

   unsigned int getX() { return posx; };
   unsigned int getY() { return posy; };

private:
   unsigned int posx, posy;
};

class Image {

public:
    Image(unsigned int nxInit, unsigned int nyInit);
    Image(unsigned int nxInit, unsigned int nyInit, pvdata_t *targetbuf);

	int clearImage();
	int fillImage(pvdata_t val);
	virtual int updateImage(float time, float dt, pvdata_t *targetbuf);

	int createRandomImage();
	int drawMultipleRandomShapes(int n_images);
	int createMultipleImages();

	int drawLine(Point2D pt1, Point2D pt2);

	int drawLine(Point2D origin,
			unsigned int length, float theta);

	int drawSquare(Point2D origin,
			unsigned int length, unsigned int theta);

	int drawSquare(Point2D pt1, Point2D pt2,
			Point2D pt3, Point2D pt4);

	int drawRectangle(Point2D origin,
			unsigned int lengtha, unsigned int lengthb,
			unsigned int theta);

	int drawRectangle(Point2D pt1, Point2D pt2,
			Point2D pt3, Point2D pt4);

	int drawQuadrilateral(Point2D pt1, Point2D pt2,
			Point2D pt3, Point2D pt4);

	int copyImage(pvdata_t *targetbuf);

	void testImage();

	void mark(unsigned int i, unsigned int j, int value);
	void mark(unsigned int i, int value);
	pvdata_t getmark(unsigned int i, unsigned int j);

	virtual int writeImageToFile(const float time,
			                     const unsigned char options);

	void setModified(bool val) { modified = val; };
	bool ifModified() { return modified; };

private:
	bool     modified;
	pvdata_t *buf;
	unsigned int nx;
	unsigned int ny;

	int writeImageToTxt(const char *filename);
	int writeImageToBin(const char *filename);
	int           drawBresenhamLine(int x0, int y0, int x1, int y1);
	void          swap(int &a, int &b);
	inline double deg2rad(int angleInDegrees);
	inline int    approx(double n);
	int           threewaytoss(double probStay,
			                   double probBack,
			                   double probForward);
};

}
#endif /* IMAGE_HPP_ */
