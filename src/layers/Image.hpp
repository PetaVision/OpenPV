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
#include "../io/imageio.hpp"
#include "../utils/cl_random.h"
#include <gdal.h>

namespace PV {

class Image : public HyPerLayer {

protected:
   Image();
   int initialize(const char * name, HyPerCol * hc, const char * filename);
   virtual int readOffsets(); // reads offsetX, offsetY from params.  Override with empty function if a derived class doesn't use these parameters (e.g. Patterns)
   static inline int calcBandWeights(int numBands, float * bandweights, GDALColorInterp * colorbandtypes);
   static inline void equalBandWeights(int numBands, float * bandweights);

   virtual bool jitter();
   virtual int calcBias(int current_bias, int step, int sizeLength);
   virtual int calcNewBiases(int stepSize);
   virtual int calcBiasedOffset(int bias, int current_offset, int step, int sizeLength);
   virtual bool calcNewOffsets(int stepSize);
   static bool constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method);
   virtual bool constrainBiases();
   virtual bool constrainOffsets();
   static double uniformRand01(uint4 * state);

public:
   Image(const char * name, HyPerCol * hc, const char * filename);
   virtual ~Image();
   virtual int initializeState();

   // primary layer interface
   //
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int neighbor);
   virtual int updateState(double time, double dt);
   virtual int outputState(double time, bool last=false);

   virtual int checkpointRead(const char * cpDir, double * timef);

   // partially override implementation of LayerDataInterface interface
   //
   const pvdata_t * getLayerData(int delay=0)   { return data; }

   virtual int  clearImage();

   float lastUpdate()  { return lastUpdateTime; }

   virtual pvdata_t * getImageBuffer() { return data; }
   virtual PVLayerLoc getImageLoc() {return imageLoc; }

   virtual int tag();

   int readImage(const char * filename);
   int readImage(const char * filename, int offsetX, int offsetY, GDALColorInterp * colorbandtypes);
   int write(const char * filename);

   int exchange();

   int toGrayScale();
   static float * convertToGrayScale(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes);

   int copyFromInteriorBuffer(float * buf, float fac);
   int copyToInteriorBuffer(unsigned char * buf, float fac);

   const char * getFilename() { return filename; }
   int getOffsetX() { return offsets[0]; }
   int getOffsetY() { return offsets[1]; }
   const int * getOffsets() { return offsets; }
   int getBiasX() { return biases[0]; }
   int getBiasY() { return biases[1]; }
   const int * getBiases() { return biases; }

private:
   int initialize_base();

protected:

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);
#endif

   //int initializeImage(const char * filename);

   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   pvdata_t * data;       // buffer containing reduced image
   char * filename;       // path to file if a file exists

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   int writeImages;      // controls writing of image file during outputState
   char * writeImagesExtension; // ".pvp", ".tif", ".png", etc.; the extension to use when writing images
   // bool useGrayScale;    // whether to convert image to grayscale
   // useGrayScale no longer used; instead setting nf=1 in params with color input images calls toGrayScale()
   int offsets[2];        // offsets array points to [offsetX, offsetY]

   bool useParamsImage;
   bool useImageBCflag;
   bool inverseFlag;
   bool normalizeLuminanceFlag;

   //float lastPhase;
   double lastUpdateTime; // time of last image update

   // Jitter parameters
   int jitterFlag;        // If true, use jitter
   int stepSize;
   int biases[2];         // When jittering, the distribution for the offset is centered on offsetX=biasX, offsetY=biasY
                          // Jittering can change the bias point on a slower timescale than the offset point changes.
   int biasConstraintMethod;  // If biases escape the bounding box, the method to coerce them into the bounding box.
   int offsetConstraintMethod; // If offsets escape the bounding box, the method to coerce them into the bounding box.
                          // The constraint method codes are 0=ignore, 1=mirror boundary conditions, 2=thresholding, 3=circular boundary conditions
   float recurrenceProb;  // If using jitter, probability that offset returns to bias position
   float persistenceProb; // If using jitter, probability that offset stays the same
   int writePosition;     // If using jitter, write positions to input/image-pos.txt
   PV_Stream * fp_pos;    // If writePosition is true, write the positions to this file
   long biasChangeTime;    // If using jitter, time period for recalculating bias position

   int jitterType;       // If using jitter, specify type of jitter (random walk or random jump)
   const static int RANDOM_WALK = 0;  // const denoting jitter is a random walk
   const static int RANDOM_JUMP = 1;  // const denoting jitter is a random jump

   uint4 rand_state;
};

}

#endif /* IMAGE_HPP_ */
