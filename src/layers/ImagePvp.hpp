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

   virtual int getFrame(double timef, double dt);
   virtual int retrieveData(double timef, double dt, int batchIndex);
   virtual int readPvp(const char * filename, int frameNumber);
   int readSparseBinaryActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber);
   int readSparseValuesActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber);
   int readNonspikingActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber);
   virtual int scatterImageFilePVP(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber);
   ImagePvp();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   long * frameStartBuf;
   int * countBuf;
   float pvpFileTime; //Current pvp file time
   long pvpFrameIdx;
   int fileNumFrames; //Number of frames
public:
   ImagePvp(const char * name, HyPerCol * hc);
   virtual ~ImagePvp();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();

   float getPvpFileTime(){ return pvpFileTime;};
   virtual long getPvpFrameIdx() { return pvpFrameIdx; }
   virtual double getDeltaUpdateTime();
private:
   int initialize_base();

   bool needFrameSizesForSpiking;
   PV_Stream * posstream;
   
}; // class ImagePvp

} // namespace PV

#endif
