/*
 * ImagePvp.hpp
 */


#ifndef IMAGEPVP_HPP_
#define IMAGEPVP_HPP_

#include "BaseInput.hpp"

namespace PV {

class ImagePvp : public BaseInput{

protected:
   /** 
    * List of parameters needed from the ImagePvp class
    * @name ImagePvp Parameters
    * @{
    */

   /**
    * @brief pvpFrameIdx: If imagePath is a pvp file, frameNumber specifies which frame to use as the image 
    */
   virtual void ioParam_pvpFrameIdx(enum ParamsIOFlag ioFlag);
   /**
    * @}
    */


   virtual int retrieveData(double timef, double dt);
   virtual int scatterImageFilePVP(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber);
   ImagePvp();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   //frameIndex is a linear index into both batches and frames
   int readPvp(const char * filename, int frameIdx, int destBatchIdx, int offsetX, int offsetY, const char* anchor);

   //char * pvpFilename;       // path to file if a file exists
   long * frameStartBuf;
   int * countBuf;
   //Current pvp file time
   float pvpFileTime;
   long pvpFrameIdx;
   //long pvpBatchIdx;
   int fileNumFrames; //Number of frames
   //int fileNumBatches;
public:
   ImagePvp(const char * name, HyPerCol * hc);
   virtual ~ImagePvp();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();

   float getPvpFileTime(){ return pvpFileTime;};
   virtual long getPvpFrameIdx() { return pvpFrameIdx; }
   //virtual long getPvpBatchIdx() { return pvpBatchIdx; }
   virtual double getDeltaUpdateTime();
private:
   int initialize_base();

   bool needFrameSizesForSpiking;
   PV_Stream * posstream;
   
}; // class ImagePvp

BaseObject * createImagePvp(char const * name, HyPerCol * hc);

} // namespace PV

#endif
