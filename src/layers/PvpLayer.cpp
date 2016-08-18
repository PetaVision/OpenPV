#include "PvpLayer.hpp"
#include "io/fileio.hpp"
#include "arch/mpi/mpi.h"

#include <assert.h>
#include <string.h>
#include <iostream>

namespace PV {

   PvpLayer::PvpLayer() {
      initialize_base();
   }

   PvpLayer::PvpLayer(const char * name, HyPerCol * hc) {
      initialize_base();
      initialize(name, hc);
   }

   PvpLayer::~PvpLayer() {
   }   

   int PvpLayer::initialize_base() {
         return PV_SUCCESS;
   }

   int PvpLayer::initialize(const char * name, HyPerCol * hc) {
      int status = InputLayer::initialize(name, hc);
      return status;
   }

   Buffer PvpLayer::retrieveData(std::string filename, int batchIndex)
   {
      pvDebug() << "RETRIEVEDATA FOR: " << filename << " ON RANK " << parent->getCommunicator()->commRank() << "\n";
      // Do I need to do anything to ensure this is rank 0?

      PV_Stream *pvpFile = pvp_open_read_file(filename.c_str(), parent->getCommunicator()); 

      // TODO: Replace this #define with a static const
      int numParams = NUM_BIN_PARAMS; 
      int params[numParams];
      double time;
      int filetype, datatype;
      pvp_read_header(pvpFile, &time, &filetype, &datatype, params, &numParams);
//      pvp_read_header(pvpFile, parent->getCommunicator(), params, &numParams);

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

      if(parent->columnId() == 0 && !mInitializedBatchIndexer) {
         mInitializedBatchIndexer = true;
         initializeBatchIndexer(params[INDEX_NBANDS]);
      }

      int frameNumber = mBatchIndexer->nextIndex(batchIndex);

      pvDebug() << "READING FRAME " << frameNumber << " FROM " << filename << "\n";
      
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
      fitBufferToLayer(result);
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
         // So we don't have to calculate frameStart and count again
         mNeedFrameSizesForSpiking = false;
      }
      long framepos = (long)mFrameStartBuffer[frameNumber];
      unsigned int length = mCountBuffer[frameNumber];
      float pvpFileTime = 0;
      PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
      PV_fread(&pvpFileTime, sizeof(double), 1, pvstream);
      unsigned int dropLength;
      PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
      assert(dropLength == length);
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
      // TODO: Does this scope change do anything meaningful? 
      {
         unsigned int dropLength;
         PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
         assert(dropLength == length);
      }
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
      int status = PV_SUCCESS;
      int recordsize = params[INDEX_RECORD_SIZE];
      int datasize = params[INDEX_DATA_SIZE];
      int framesize = recordsize*datasize+8;
      long framepos = (long)framesize * (long)frameNumber + (long)params[INDEX_HEADER_SIZE];
      //ONLY READING TIME INFO HERE
      status = PV_fseek(pvstream, framepos, SEEK_SET);
      if (status != 0) {
         pvErrorNoExit().printf("scatterImageFilePVP: Unable to seek to start of frame %d in \"%s\": %s\n", frameNumber, pvstream->name, strerror(errno));
         status = PV_FAILURE;
      }
      if (status == PV_SUCCESS) {
         float pvpFileTime = 0;
         size_t numread = PV::PV_fread(&pvpFileTime, sizeof(double), (size_t) 1, pvstream);
         if (numread != (size_t) 1) {
            pvErrorNoExit(errorMessage); //TODO: What is errorMessage? Where does it come from?
            errorMessage.printf("scatterImageFilePVP: Unable to read timestamp from frame %d of file \"%s\":", frameNumber, pvstream->name);
            if (feof(pvstream->fp)) {
               errorMessage.printf(" end-of-file."); 
            }
            if (ferror(pvstream->fp)) {
               errorMessage.printf(" fread error."); 
            }
            errorMessage.printf("\n");
            status = PV_FAILURE;
         }
      }
      // Assumes that imageData is of type float.
      int width      = params[INDEX_NX];
      int height     = params[INDEX_NY];
      int features   = params[INDEX_NF];
      std::vector<float> data(width * height * features);
      PV_fread(data.data(), sizeof(float), (size_t) (width * height * features), pvstream);
      return Buffer(data, width, height, features);
   }

} // namespace PV
