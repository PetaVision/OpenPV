/*
 * ImageCreator.hpp
 *
 *  Created on: Aug 25, 2009
 *      Author: Shreyas
 */

#ifndef IMAGECREATOR_HPP_
#define IMAGECREATOR_HPP_

#include "Image.hpp"
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

class ImageCreator : public Image {

public:
    ImageCreator(const char * name, HyPerCol * hc);
    virtual ~ImageCreator();

    int initialize();

    int clearImage();
	int fillImage(pvdata_t val);
	virtual bool updateImage(float time, float dt);

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
	unsigned char getmark(unsigned int i, unsigned int j);

	virtual int writeImageToFile(const float time,
			                     const unsigned char options);

	void setModified(bool val) { modified = val; };
	bool ifModified() { return modified; };

private:
	HyPerCol * hc;
	bool modified;

	unsigned char * drawBuffer;

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
#endif /* IMAGECREATOR_HPP_ */
