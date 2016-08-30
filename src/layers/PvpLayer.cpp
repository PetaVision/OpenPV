#include "PvpLayer.hpp"
#include "io/fileio.hpp"
#include "arch/mpi/mpi.h"

#include <cstring>
#include <iostream>

namespace PV {

   PvpLayer::PvpLayer(const char * name, HyPerCol * hc) {
      initialize(name, hc);
   }

   PvpLayer::~PvpLayer() {
   }   

   int PvpLayer::allocateDataStructures() {
      int status = InputLayer::allocateDataStructures();
      if(status != PV_SUCCESS) {
         return status;
      }
      if(parent->columnId() == 0) {
         pvErrorIf(getUsingFileList(),
            "%s: PvpLayer does not support using a list of files.\n",
            getName());
         initializeBatchIndexer(mPvpFrameCount);
      }
      return status;
   }

   Buffer PvpLayer::retrieveData(std::string filename, int batchIndex) {
      PV_Stream *pvpFile = pvp_open_read_file(filename.c_str(), parent->getCommunicator()); 
      pvErrorIf(pvpFile == nullptr, "Could not open %s\n", filename.c_str());

      int numParams = NUM_BIN_PARAMS; 
      int params[numParams];
      double time;
      int filetype, datatype;
      pvp_read_header(pvpFile, &time, &filetype, &datatype, params, &numParams);

      // TODO: All of this pvp loading code should live somewhere else, maybe as file i/o for Buffer?
      PVLayerLoc pvpInfo;
      
      pvpInfo.nx = params[INDEX_NX];
      pvpInfo.ny = params[INDEX_NY];
      pvpInfo.nf = params[INDEX_NF];
      pvpInfo.nxGlobal = params[INDEX_NX_GLOBAL];
      pvpInfo.nyGlobal = params[INDEX_NY_GLOBAL];
      pvpInfo.kx0 = params[INDEX_KX0];
      pvpInfo.ky0 = params[INDEX_KY0];
      
      int nxProcs = params[INDEX_NX_PROCS];
      int nyProcs = params[INDEX_NY_PROCS];
      
      bool invalidFile =
            pvpInfo.nx != pvpInfo.nxGlobal
         || pvpInfo.ny != pvpInfo.nyGlobal
         || nxProcs != 1
         || nyProcs != 1
         || pvpInfo.kx0 != 0
         || pvpInfo.ky0 != 0;

      pvErrorIf(invalidFile,
            "File \"%s\" appears to be an obsolete version of the .pvp format.\n", filename.c_str());

      // This is present so that when nextInput() is called during
      // InputLayer::allocateDataStructures, we correctly load the
      // inital state of the layer. Then, after InputLayer::allocate
      // is finished, PvpLayer::allocate reinitializes the BatchIndexer
      // so that the first update state does not skip the first
      // frame in the batch.
      if(mPvpFrameCount == -1) {
         mPvpFrameCount = params[INDEX_NBANDS];
         initializeBatchIndexer(mPvpFrameCount);
      }

      int frameNumber = 0;

      // If we're playing through the pvp file like a movie, use
      // BatchIndexer to get the frame number. Otherwise, just use
      // the start_frame_index value for this batch.
      if(getDisplayPeriod() > 0) {
         frameNumber = mBatchIndexer->nextIndex(batchIndex);
      }
      else {
         frameNumber = getStartIndex(batchIndex);
      }

      int fileType = params[INDEX_FILE_TYPE];
      Buffer result;
      switch(fileType) {
         case PVP_ACT_FILE_TYPE:
            result = readSparseBinaryActivityFrame(numParams, params, pvpFile, frameNumber);
            break;
         case PVP_ACT_SPARSEVALUES_FILE_TYPE:
            result = readSparseValuesActivityFrame(numParams, params, pvpFile, frameNumber);
            break;
         case PVP_NONSPIKING_ACT_FILE_TYPE:
            result = readNonspikingActivityFrame(numParams, params, pvpFile, frameNumber);
            break;
         default:
            pvp_close_file(pvpFile, parent->getCommunicator());
            pvError() << "Unrecognized or unsupported .pvp file format: " << fileType << std::endl;
            break;
      }

      pvp_close_file(pvpFile, parent->getCommunicator());
      return result;
   }

   Buffer PvpLayer::readSparseBinaryActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {

      // Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
      // Only need to do this once
      int status = PV_SUCCESS;
      if (mNeedFrameSizesForSpiking) { 
         pvInfo() << "Calculating file positions\n";
         mFrameStartBuffer.clear();
         mFrameStartBuffer.resize(params[INDEX_NBANDS]);
         mCountBuffer.clear();
         mCountBuffer.resize(params[INDEX_NBANDS]);

         // fseek past the header and first timestamp
         PV_fseek(pvstream, (long)8 + (long)params[INDEX_HEADER_SIZE], SEEK_SET);
         int percent = 0;
         for (int f = 0; f < params[INDEX_NBANDS]; f++) {
            int newpercent = 100*(f/(params[INDEX_NBANDS]));
            if(percent != newpercent) {
               percent = newpercent;
               pvInfo() << "\r" << percent << "\% Done";
               pvInfo().flush();
            }
            // TODO: Where does this magic number come from?
            // First byte position should always be 92
            if (f == 0) {
               mFrameStartBuffer[f] = (long)92;
            }
            else {
               // Read in the number of active neurons for that frame and calculate byte position
               PV_fread(&mCountBuffer[f-1], sizeof(int), 1, pvstream);
               mFrameStartBuffer[f] = mFrameStartBuffer[f-1] + (long)mCountBuffer[f-1]*(long)params[INDEX_DATA_SIZE] + (long)12;
               PV::PV_fseek(pvstream, mFrameStartBuffer[f] - (long)4, SEEK_SET);
            }
         }
         pvInfo() << "\r" << percent << "% Done\n";
         pvInfo().flush();

         // We still need the last count
         PV_fread(&mCountBuffer[(params[INDEX_NBANDS])-1], sizeof(int), 1, pvstream);
         mNeedFrameSizesForSpiking = false;
      }

      long framepos = (long)mFrameStartBuffer[frameNumber];
      unsigned int length = mCountBuffer[frameNumber];
      float pvpFileTime = 0;
      PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
      PV_fread(&pvpFileTime, sizeof(double), 1, pvstream);
      unsigned int dropLength;
      PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
      pvAssert(dropLength == length);
      int locations[length];
      PV_fread(locations, sizeof(int), length, pvstream);
      int width      = params[INDEX_NX];
      int height     = params[INDEX_NY];
      int features   = params[INDEX_NF];
      std::vector<float> data(width * height * features);
      for(unsigned int l = 0; l < length; l++) {
         data.at(locations[l]) = 1.0f;
      }
      return Buffer(data, width, height, features);
   }

   Buffer PvpLayer::readSparseValuesActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {
      
      // Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
      // Only need to do this once
      if (mNeedFrameSizesForSpiking) {
         pvInfo() << "Calculating file positions\n";
         mFrameStartBuffer.clear();
         mFrameStartBuffer.resize(params[INDEX_NBANDS]);
         mCountBuffer.clear();   
         mCountBuffer.resize(params[INDEX_NBANDS]);

         // Fseek past the header and first timestamp
         PV_fseek(pvstream, (long)8 + (long)params[INDEX_HEADER_SIZE], SEEK_SET);
         int percent = 0;
         for (int f = 0; f<params[INDEX_NBANDS]; f++) {
            int newpercent = 100*(f/(params[INDEX_NBANDS]));
            if(percent != newpercent) {
               percent = newpercent;
               pvInfo() << "\r" << percent << "\% Done";
               pvInfo().flush();
            }

            // First byte position should always be 92
            if(f == 0) {
               mFrameStartBuffer[f] = (long)92;
            }
            else {
               // Read in the number of active neurons for that frame and calculate byte position
               PV_fread(&mCountBuffer[f-1], sizeof(int), 1, pvstream);
               mFrameStartBuffer[f] = mFrameStartBuffer[f-1] + (long)mCountBuffer[f-1]*(long)params[INDEX_DATA_SIZE] + (long)12;
               PV_fseek(pvstream, mFrameStartBuffer[f] - (long)4, SEEK_SET);
            }
         }
         pvInfo() << "\r" << percent << "% Done\n";
         pvInfo().flush();

         // We still need the last count
         PV_fread(&mCountBuffer[(params[INDEX_NBANDS])-1], sizeof(int), 1, pvstream);

         // So we don't have to calculate frameStart and count again
         mNeedFrameSizesForSpiking = false;
      }

      long framepos = (long)mFrameStartBuffer[frameNumber];
      unsigned int length = mCountBuffer[frameNumber];
      float pvpFileTime = 0;
      PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
      PV_fread(&pvpFileTime, sizeof(double), 1, pvstream);
      unsigned int dropLength;
      PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
      assert(dropLength == length);
      struct locvalue {
         int location;
         float value;
      };
      struct locvalue locvalues[length];
      PV_fread(locvalues, sizeof(locvalue), length, pvstream);
      int width      = params[INDEX_NX];
      int height     = params[INDEX_NY];
      int features   = params[INDEX_NF];
      std::vector<float> data(width * height * features);
      for(unsigned int l = 0; l < length; l++) {
         data.at(locvalues[l].location) = locvalues[l].value;
      }
      return Buffer(data, width, height, features);
   }

   Buffer PvpLayer::readNonspikingActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {
      int recordsize = params[INDEX_RECORD_SIZE];
      int datasize   = params[INDEX_DATA_SIZE];
      int framesize  = recordsize * datasize + 8; // What is this magic number?
      long framepos  = (long)framesize * (long)frameNumber + (long)params[INDEX_HEADER_SIZE];
      //ONLY READING TIME INFO HERE
      if (PV_fseek(pvstream, framepos, SEEK_SET) == PV_FAILURE) {
         pvError().printf("scatterImageFilePVP: Unable to seek to start of frame %d in \"%s\": %s\n", frameNumber, pvstream->name, strerror(errno));
      }
      else { 
         float pvpFileTime = 0;
         size_t numRead = PV::PV_fread(&pvpFileTime, sizeof(double), (size_t) 1, pvstream);
         if (numRead != (size_t) 1) {
            pvErrorNoExit(errorMessage); //TODO: What is errorMessage? Where does it come from?
            pvError().printf("scatterImageFilePVP: Unable to read timestamp from frame %d of file \"%s\": %s", frameNumber, pvstream->name, feof(pvstream->fp) ? " EOF" : " fread error");
         }
      }
      // Assumes that imageData is of type float.
      int width      = params[INDEX_NX];
      int height     = params[INDEX_NY];
      int features   = params[INDEX_NF];
      std::vector<float> data(width * height * features);
      size_t numRead = PV_fread(data.data(), sizeof(float), data.size(), pvstream);
      pvErrorIf(numRead != data.size(), "Expected to read %d values, found %d instead.\n", data.size(), numRead);
      return Buffer(data, width, height, features);
   }
}
