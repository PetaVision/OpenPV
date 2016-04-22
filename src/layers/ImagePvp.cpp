/*
 * ImagePvp.cpp
 */

#include "ImagePvp.hpp"

namespace PV {

ImagePvp::ImagePvp() {
   initialize_base();
}

ImagePvp::ImagePvp(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ImagePvp::~ImagePvp() {
   if(frameStartBuf){
      free(frameStartBuf);
      frameStartBuf = NULL;
   }
   if(countBuf){
      free(countBuf);
      countBuf = NULL;
   }
}

int ImagePvp::initialize_base() {
   needFrameSizesForSpiking = true;
   frameStartBuf = NULL;
   countBuf = NULL;
   //pvpFilename = NULL;
   pvpFrameIdx = 0;
   //pvpBatchIdx = 0;
   return PV_SUCCESS;
}

int ImagePvp::initialize(const char * name, HyPerCol * hc) {
   int status = BaseInput::initialize(name, hc);

   PV_Stream * pvstream = NULL;
   if (getParent()->icCommunicator()->commRank()==0) {
      pvstream = PV::PV_fopen(inputPath, "rb", false/*verifyWrites*/);
   }
   int numParams = NUM_PAR_BYTE_PARAMS;
   int params[numParams];
   pvp_read_header(pvstream, getParent()->icCommunicator(), params, &numParams);
   PV::PV_fclose(pvstream); pvstream = NULL;
   if (numParams != NUM_BIN_PARAMS || params[INDEX_HEADER_SIZE] != NUM_BIN_PARAMS*sizeof(int) || params[INDEX_NUM_PARAMS] != NUM_BIN_PARAMS) {
      std::cout << "ImagePvp:: inputPath \"" << inputPath << "\" is not a .pvp file.\n";
      exit(EXIT_FAILURE);
   }
   fileNumFrames = params[INDEX_NBANDS]; 
   //fileNumBatches = params[INDEX_NBATCH];
   
   if(pvpFrameIdx < 0 || pvpFrameIdx >= fileNumFrames){
      std::cout << "ImagePvp:: pvpFrameIndex of " << pvpFrameIdx << " out of bounds, file contains " << fileNumFrames << " frames\n";
      exit(EXIT_FAILURE);
   }

   return status;
}

int ImagePvp::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);
   //ioParam_pvpPath(ioFlag);
   ioParam_pvpFrameIdx(ioFlag);
   //ioParam_pvpBatchIdx(ioFlag);
   return status;
}

//void ImagePvp::ioParam_inputPath(enum ParamsIOFlag ioFlag) {
//   parent->ioParamStringRequired(ioFlag, name, "inputPath", &pvpFilename);
//}

void ImagePvp::ioParam_pvpFrameIdx(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "pvpFrameIdx", &pvpFrameIdx, pvpFrameIdx);
}

//void ImagePvp::ioParam_pvpBatchIdx(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "pvpBatchIdx", &pvpBatchIdx, pvpBatchIdx);
//   //If the file specifies a "dont care" numBatches, but the user specifies a pvpbatchIdx, error
//   if(pvpBatchIdx > 0 && fileNumBatches == 0){
//      std::cout << "ImagePvp:: File " << inputPath << " does not contain a numBatches, cannot specify a pvpBatchIdx\n";
//      exit(-1);
//   }
//   //Check bounds
//   if(pvpBatchIdx < 0 || pvpBatchIdx >= fileNumBatches){
//      std::cout << "ImagePvp:: pvpBatchIdx of " << pvpBatchIdx << " out of bounds, file contains " << fileNumBatches << " batches\n";
//      exit(-1);
//   }
//}

int ImagePvp::communicateInitInfo() {
   int status = BaseInput::communicateInitInfo();
   int fileType = getFileType(inputPath);
   if(fileType != PVP_FILE_TYPE){
      std::cout << "ImagePvp/MoviePvp reads PVP files. Use Image/Movie for reading images.\n";
      exit(-1);
   }
   return status;
}


//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int ImagePvp::retrieveData(double timef, double dt)
{
   int status = PV_SUCCESS;
   for(int b = 0; b < parent->getNBatch(); b++){
      status = readPvp(inputPath, pvpFrameIdx, b, offsets[0], offsets[1], offsetAnchor);
      assert(status == PV_SUCCESS);
   }
   return status;
}

double ImagePvp::getDeltaUpdateTime(){
   if(jitterFlag){
      return parent->getDeltaTime();
   }
   else{
      return -1; //Never update
   }
}

int ImagePvp::readPvp(const char * filename, int frameIdx, int destBatchIdx, int offsetX, int offsetY, const char* anchor)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   if(useImageBCflag){ //Expand dimensions to the extended space
      loc->nx = loc->nx + loc->halo.lt + loc->halo.rt;
      loc->ny = loc->ny + loc->halo.dn + loc->halo.up;
      //TODO this seems to fix the pvp ext vs res offset if imageBC flag is on, but only for no mpi runs
      //offsetX = offsetX - loc->halo.lt;
      //offsetY = offsetY - loc->halo.up;
   }

   // read the image and scatter the local portions
   bool usingTempFile = false;
   int numAttempts = 5;

   status = getImageInfoPVP(filename, parent->icCommunicator(), &imageLoc);
   if(status != 0) {
      fprintf(stderr, "Movie: Unable to get image info for \"%s\"\n", filename);
      abort();
   }

   //See if we are padding
   int n = loc->nx * loc->ny * imageLoc.nf;

   // Use number of bands in file instead of in params, to allow for grayscale conversion
   float * buf = new float[n];
   assert(buf != NULL);

   int aOffsetX = getOffsetX(anchor, offsetX);
   int aOffsetY = getOffsetY(anchor, offsetY);

   status = scatterImageFilePVP(filename, aOffsetX, aOffsetY, parent->icCommunicator(), loc, buf, frameIdx);
   if (status != PV_SUCCESS) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Image::readImage failed for layer \"%s\"\n", getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if( status == PV_SUCCESS ) copyFromInteriorBuffer(buf, destBatchIdx, 1.0f);

   delete[] buf;

   if(useImageBCflag){ //Restore non-extended dimensions
      loc->nx = loc->nx - loc->halo.lt - loc->halo.rt;
      loc->ny = loc->ny - loc->halo.dn - loc->halo.up;
   }

   return status;
}

int ImagePvp::scatterImageFilePVP(const char * filename, int xOffset, int yOffset,
                        PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber)
{

   //// Read a PVP file and scatter it to the multiple processes.
   //if (autoResizeFlag) {
   //   if (parent->columnId()==0) {
   //      fprintf(stderr, "%s \"%s\" error: autoRescaleFlag=true has not been implemented for .pvp files.\n",
   //         getKeyword(), name);
   //   }
   //   MPI_Barrier(parent->icCommunicator()->communicator());
   //   exit(EXIT_FAILURE);
   //}
   
   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = comm->commRank();
   PV_Stream * pvstream;
   if(rank==rootproc){
       pvstream = PV::pvp_open_read_file(filename, comm);
   }
   else{
       pvstream = NULL;
   }
   long length = 0;
   int numParams = NUM_BIN_PARAMS;
   int params[numParams];
   PV::pvp_read_header(pvstream, comm, params, &numParams);
   if (frameNumber < 0 || frameNumber >= fileNumFrames) {
      if (rank==rootproc) {
         fprintf(stderr, "scatterImageFilePVP error: requested frameNumber %d but file \"%s\" only has frames numbered 0 through %d.\n", frameNumber, filename, params[INDEX_NBANDS]-1);
      }
      return PV_FAILURE;
   }

   if (rank==rootproc) {
      PVLayerLoc fileloc;
      int headerSize = params[INDEX_HEADER_SIZE];
      fileloc.nx = params[INDEX_NX];
      fileloc.ny = params[INDEX_NY];
      fileloc.nf = params[INDEX_NF];
      //fileloc.nbatch = params[INDEX_NBATCH];
      fileloc.nxGlobal = params[INDEX_NX_GLOBAL];
      fileloc.nyGlobal = params[INDEX_NY_GLOBAL];
      fileloc.kx0 = params[INDEX_KX0];
      fileloc.ky0 = params[INDEX_KY0];
      int nxProcs = params[INDEX_NX_PROCS];
      int nyProcs = params[INDEX_NY_PROCS];
      int recordsize = params[INDEX_RECORD_SIZE];
      int datasize = params[INDEX_DATA_SIZE];
      if (fileloc.nx != fileloc.nxGlobal || fileloc.ny != fileloc.nyGlobal ||
          nxProcs != 1 || nyProcs != 1 ||
          fileloc.kx0 != 0 || fileloc.ky0 != 0) {
          fprintf(stderr, "File \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
          abort();
      }

      bool spiking = false;
      double timed = 0.0;
      int filetype = params[INDEX_FILE_TYPE];
      int framesize;
      long framepos;
      unsigned len;
      char tempPosFilename [PV_PATH_MAX];
      const char * posFilename;
      int targetIdx = 0;

      //int nbatches = params[INDEX_NBATCH];
      //if(nbatches == 0){
      //   nbatches = 1;
      //}

      switch (filetype) {
      case PVP_FILE_TYPE:
         break;

      case PVP_ACT_FILE_TYPE:
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:

         //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
         //Only need to do this once

         //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
         //Only need to do this once
         if (needFrameSizesForSpiking) {
            std::cout << "Calculating file positions\n";
            frameStartBuf = (long *) calloc(params[INDEX_NBANDS] ,sizeof(long));
            if (frameStartBuf==NULL) {
               fprintf(stderr, "scatterImageFilePVP unable to allocate memory for frameStart.\n");
               status = PV_FAILURE;
               abort();
            }

            countBuf = (int *) calloc(params[INDEX_NBANDS] ,sizeof(int));
            if (countBuf==NULL) {
               fprintf(stderr, "scatterImageFilePVP unable to allocate memory for frameLength.\n");
               status = PV_FAILURE;
               abort();
            }

            //Fseek past the header and first timestamp
            PV::PV_fseek(pvstream, (long)8 + (long)headerSize, SEEK_SET);
            int percent = 0;
            for (int f = 0; f<params[INDEX_NBANDS]; f++) {
               int newpercent = 100*(f/(params[INDEX_NBANDS]));
               if(percent != newpercent){
                  percent = newpercent;
                  std::cout << "\r" << percent << "\% Done";
                  std::cout.flush();
               }
               //First byte position should always be 92
               if (f == 0) {
                  frameStartBuf[f] = (long)92;
               }
               //Read in the number of active neurons for that frame and calculate byte position
               else {
                  PV::PV_fread(&countBuf[f-1], sizeof(int), 1, pvstream);
                  frameStartBuf[f] = frameStartBuf[f-1] + (long)countBuf[f-1]*(long)datasize + (long)12;
                  PV::PV_fseek(pvstream, frameStartBuf[f] - (long)4, SEEK_SET);
               }
            }
            std::cout << "\r" << percent << "% Done\n";
            std::cout.flush();
            //We still need the last count
            PV::PV_fread(&countBuf[(params[INDEX_NBANDS])-1], sizeof(int), 1, pvstream);

            //So we don't have to calculate frameStart and count again
            needFrameSizesForSpiking = false;
         }

         framepos = (long)frameStartBuf[frameNumber];
         length = countBuf[frameNumber];
         PV::PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
         PV::PV_fread(&timed, sizeof(double), 1, pvstream);
         unsigned int dropLength;
         PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
         assert(dropLength == length);
         status = PV_SUCCESS;
         break;
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         length = 0; //Don't need to compute this for nonspiking files.
         framesize = recordsize*datasize*nxProcs*nyProcs+8;
         framepos = (long)framesize * (long)frameNumber + (long)headerSize;
         //ONLY READING TIME INFO HERE
         status = PV::PV_fseek(pvstream, framepos, SEEK_SET);
         if (status != 0) {
            fprintf(stderr, "scatterImageFilePVP error: Unable to seek to start of frame %d in \"%s\": %s\n", frameNumber, filename, strerror(errno));
            status = PV_FAILURE;
         }
         if (status == PV_SUCCESS) {
            size_t numread = PV::PV_fread(&timed, sizeof(double), (size_t) 1, pvstream);
            if (numread != (size_t) 1) {
               fprintf(stderr, "scatterImageFilePVP error: Unable to read timestamp from frame %d of file \"%s\":", frameNumber, filename);
               if (feof(pvstream->fp)) { fprintf(stderr, " end-of-file."); }
               if (ferror(pvstream->fp)) { fprintf(stderr, " fread error."); }
               fprintf(stderr, "\n");
               status = PV_FAILURE;
            }
         }
         break;
      case PVP_WGT_FILE_TYPE:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": file is a weight file, not an image file.\n", filename);
         break;
      case PVP_KERNEL_FILE_TYPE:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": file is a weight file, not an image file.\n", filename);
         break;

      default:
         fprintf(stderr, "scatterImageFilePVP error opening \"%s\": filetype %d is unrecognized.\n", filename ,filetype);
         status = PV_FAILURE;
         break;
      }
      if (status != PV_SUCCESS) {
         exit(EXIT_FAILURE);
      }

      pvpFileTime = timed;
      //This is being printed twice: track down
      //std::cout << "Rank " << rank << " Reading pvpFileTime " << pvpFileTime << " at timestep " << parent->simulationTime() << " with offset (" << xOffset << "," << yOffset << ")\n";

      scatterActivity(pvstream, comm, rootproc, buf, loc, false, &fileloc, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      // buf is a nonextended layer.  Image layers copy the extended buffer data into buf by calling Image::copyToInteriorBuffer
      PV::PV_fclose(pvstream); pvstream = NULL;
   }
   else {
      if ((params[INDEX_FILE_TYPE] == PVP_ACT_FILE_TYPE) || (params[INDEX_FILE_TYPE] == PVP_ACT_SPARSEVALUES_FILE_TYPE)) {
         scatterActivity(pvstream, comm, rootproc, buf, loc, false, NULL, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      }

      else if (params[INDEX_FILE_TYPE] == PVP_NONSPIKING_ACT_FILE_TYPE) {
         scatterActivity(pvstream, comm, rootproc, buf, loc, false, NULL, xOffset, yOffset, params[INDEX_FILE_TYPE], length);
      }
   }
   return status;
}

BaseObject * createImagePvp(char const * name, HyPerCol * hc) {
   return hc ? new ImagePvp(name, hc) : NULL;
}

}
