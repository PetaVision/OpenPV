/*
 * TextStream.hpp
 *
 *  Created on: May 6, 2013
 *      Author: dpaiton
 */

#ifndef TEXTSTREAM_HPP_
#define TEXTSTREAM_HPP_

#include <layers/HyPerLayer.hpp>
#include <columns/HyPerCol.hpp>

#ifndef STAT_H
#include <sys/stat.h>
#endif

namespace PVtext {

class TextStream : public PV::HyPerLayer{

public:
   TextStream(const char * name, PV::HyPerCol * hc);
   virtual ~TextStream();
   virtual int allocateDataStructures();
   //virtual bool needUpdate(double time, double dt);
   virtual int updateState(double time, double dt);
   //TODO Is this being used?
   //float lastUpdate()  { return lastUpdateTime; }
   virtual bool activityIsSpiking() { return false; }

private:
   int initialize_base();
   int encodedChar;

protected:
   TextStream();
   int initialize(const char * name, PV::HyPerCol * hc);
   int getCharEncoding(const unsigned char * printableASCIIChar);
   char getCharType(int encodedChar);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxScale(enum ParamsIOFlag ioFlag); // Override from HyPerLayer - will just set nxScale now instead of reading
   virtual void ioParam_nyScale(enum ParamsIOFlag ioFlag); // Override from HyPerLayer - will just set nyScale now instead of reading
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);      // Override from HyPerLayer - will just set NF now instead of reading
   virtual void ioParam_useCapitalization(enum ParamsIOFlag ioFlag);
   virtual void ioParam_loopInput(enum ParamsIOFlag ioFlag);
   virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_textInputPath(enum ParamsIOFlag ioFlag);
   virtual void ioParam_textOffset(enum ParamsIOFlag ioFlag);
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_textBCFlag(enum ParamsIOFlag ioFlag);
   //TestStream does not need trigger flag, since it's overwriting needUpdate
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {}
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {}
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

   virtual double getDeltaUpdateTime();
   
   virtual int allocateV();
   virtual int setActivity();

   int scatterTextBuffer(PV::Communicator * comm, const PVLayerLoc * loc);
   int readFileToBuffer(int offset, const PVLayerLoc * loc, int * buf);
   int loadBufferIntoData(const PVLayerLoc * loc, int * buf);


   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   PV_Stream * fileStream;

   char * filename;          // Path to file if a file exists

   PVLayerLoc textLoc;       // Size/location of actual image in global context
   pvdata_t * textData;      // Buffer containing image

   double displayPeriod;     // Length of time a string 'frame' is displayed
   //double nextDisplayTime;

   //lastUpdateTime already exists in HyPerLayer
   //double lastUpdateTime;    // Time of last image update

   int textOffset;           // Starting point for run

   bool useCapitalization;   // Should mapping account for capital letters
   bool loopInput;           // Should the algorithm loop through the text file until specified total run time is completed or exit gracefully
   bool textBCFlag;          // Grab text in either direction
};
}

#endif /* TEXTSTREAM_HPP_ */
