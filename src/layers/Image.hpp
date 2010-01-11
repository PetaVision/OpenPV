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

   virtual int initialize_base(const char * name, HyPerCol * hc);
   virtual int initialize_data(const LayerLoc * loc);

   virtual bool updateImage(float time, float dt);
   virtual int clearImage();

   float lastUpdate()  { return lastUpdateTime; }

   virtual pvdata_t * getImageBuffer();
   virtual LayerLoc   getImageLoc();

   int read(const char * filename);
   int write(const char * filename);

   int exchange();

   int toGrayScale();
   static int convertToGrayScale(LayerLoc * loc, unsigned char * buf);

   int  convolve(int width);
   void setTau(int t)                { tau = t; }

   int copyFromInteriorBuffer(const unsigned char * buf);
   int copyToInteriorBuffer(unsigned char * buf);

protected:

   char * name;          // the name of the image object

   Communicator * comm;           // the communicator object for reading/writing files
   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   LayerLoc loc;          // size/location of layer
   pvdata_t * data;       // buffer containing reduced image

   LayerLoc   imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   float lastPhase;
   float lastUpdateTime; // time of last image update

   float tau;
};

}

#endif /* IMAGE_HPP_ */
