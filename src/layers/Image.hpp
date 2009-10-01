/*
 * Image.hpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "Image.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class Image {

protected:
   Image(const char * name, HyPerCol * hc);

public:
   Image(const char * name, HyPerCol * hc, const char * filename);
   virtual ~Image();

   int init_base(const char * name, HyPerCol * hc);

   virtual bool updateImage(float time, float dt);

   virtual pvdata_t * getImageBuffer();
   virtual LayerLoc   getImageLoc();

   int read(const char * filename);
   int write(const char * filename);

   static
   int convertToGrayScale(LayerLoc * loc, unsigned char * buf);

   int toGrayScale();

   int  convolution();
   void setTau(int t)             { tau = t; }

protected:

   int copyFromInteriorBuffer(const unsigned char * buf);
   int copyToInteriorBuffer(unsigned char * buf);

   char * name;          // the name of the image object

   Communicator * comm;  // the communicator object for reading/writing files
   LayerLoc loc;         // size/location of image
   pvdata_t *  data;     // buffer containing image
   float tau;
};

}

#endif /* IMAGE_HPP_ */
