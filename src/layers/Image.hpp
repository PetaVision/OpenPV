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
   Image(const char * name, HyPerCol * hc);

public:
   Image(const char * name, HyPerCol * hc, const char * filename);
   virtual ~Image();

#ifdef OBSOLETE
   virtual int initialize_base(const char * name, HyPerCol * hc);
#endif
   virtual int initializeImage(const char * filename);

   // primary layer interface
   //
   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
//   virtual int publish(InterColComm * comm, float time);
   virtual int outputState(float time, bool last=false);

#ifdef OBSOLETE
   // implementation of LayerDataInterface interface
   //
   const PVLayerLoc * getLayerLoc()  { return clayer->loc; /*return &loc;*/ }
   const pvdata_t * getLayerData()   { return getImageBuffer(); }
   bool isExtended()                 { return true; }

   virtual bool updateImage(float time, float dt);
#endif
   virtual int  clearImage();

   float lastUpdate()  { return lastUpdateTime; }

   virtual pvdata_t * getImageBuffer();
   virtual PVLayerLoc getImageLoc();

   virtual int tag();

   int read(const char * filename);
   int write(const char * filename);

   int exchange();

   int toGrayScale();
   static int convertToGrayScale(PVLayerLoc * loc, unsigned char * buf);

   int  convolve(int width);
   void setTau(float t)                { tau = t; }

   int copyFromInteriorBuffer(const unsigned char * buf);
   int copyToInteriorBuffer(unsigned char * buf);
   int gatherToInteriorBuffer(unsigned char * buf);

protected:

   //int initializeImage(const char * filename);

#ifdef OBSOLETE
   Communicator * comm;           // the communicator object for reading/writing files
#endif
   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

#ifdef OBSOLETE
   PVLayerLoc loc;        // size/location of layer
#endif
   pvdata_t * data;       // buffer containing reduced image
   char * filename; // path to file if a file exists

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   float lastPhase;
   float lastUpdateTime; // time of last image update

   float tau;
};

}

#endif /* IMAGE_HPP_ */
