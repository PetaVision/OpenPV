/*
 * Image.hpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class Image : public HyPerLayer {

protected:
   Image();
   int initialize(const char * name, HyPerCol * hc, const char * filename);
   virtual int readOffsets(); // reads offsetX, offsetY from params.  Override with empty function if a derived class doesn't use these parameters (e.g. Patterns)
   virtual int initializeV();

public:
   Image(const char * name, HyPerCol * hc, const char * filename);
   virtual ~Image();

   // primary layer interface
   //
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
   virtual int outputState(float time, bool last=false);

   virtual int checkpointRead(float * timef);

   // partially override implementation of LayerDataInterface interface
   //
   const pvdata_t * getLayerData(int delay=0)   { return data; }

   virtual int  clearImage();

   float lastUpdate()  { return lastUpdateTime; }

   virtual pvdata_t * getImageBuffer() { return data; }
   virtual PVLayerLoc getImageLoc() {return imageLoc; }

   virtual int tag();

   int readImage(const char * filename);
   int readImage(const char * filename, int offsetX, int offsetY);
   int write(const char * filename);

   int exchange();

   int toGrayScale();
   static float * convertToGrayScale(float * buf, int nx, int ny, int numBands);

   int  convolve(int width);
   // void setTau(float t)                { tau = t; }

   int copyFromInteriorBuffer(float * buf, float fac);
   int copyToInteriorBuffer(unsigned char * buf, float fac);

   const char * getFilename() { return filename; }
   int getOffsetX() { return offsetX; }
   int getOffsetY() { return offsetY; }

private:
   int initialize_base();

protected:

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();
#endif

   //int initializeImage(const char * filename);

   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   pvdata_t * data;       // buffer containing reduced image
   char * filename;       // path to file if a file exists

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   int writeImages;      // controls writing of image file during outputState
   // bool useGrayScale;    // whether to convert image to grayscale
   // useGrayScale no longer used; instead setting nf=1 in params with color input images calls toGrayScale()
   int offsetX;          // offset of layer section within full image frame
   int offsetY;

   bool useParamsImage;

   float lastPhase;
   float lastUpdateTime; // time of last image update

   // float tau;  // tau is not used by image or any subclasses
};

}

#endif /* IMAGE_HPP_ */
