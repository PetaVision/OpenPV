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
   pvpFrameIdx = 0;
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
      pvError() << "ImagePvp:: inputPath \"" << inputPath << "\" is not a .pvp file.\n";
   }
   fileNumFrames = params[INDEX_NBANDS]; 
   
   if(pvpFrameIdx < 0 || pvpFrameIdx >= fileNumFrames){
      pvError() << "ImagePvp:: pvpFrameIndex of " << pvpFrameIdx << " out of bounds, file contains " << fileNumFrames << " frames\n";
   }

   return status;
}

int ImagePvp::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);
   ioParam_pvpFrameIdx(ioFlag);
   return status;
}

void ImagePvp::ioParam_pvpFrameIdx(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "pvpFrameIdx", &pvpFrameIdx, pvpFrameIdx);
}

int ImagePvp::communicateInitInfo() {
   int status = BaseInput::communicateInitInfo();
   int fileType = getFileType(inputPath);
   if(fileType != PVP_FILE_TYPE){
      pvError() << "ImagePvp/MoviePvp reads PVP files. Use Image/Movie for reading images.\n";
   }
   return status;
}

int ImagePvp::getFrame(double timef, double dt) {
   int status = PV_SUCCESS;
   for(int b = 0; b < parent->getNBatch(); b++) {
      if (status == PV_SUCCESS) { status = retrieveData(timef, dt, b); }
      if (status == PV_SUCCESS) { status = scatterInput(b); }
   }
   if (status == PV_SUCCESS) { status = postProcess(timef, dt); }
   return status;
}

//Image readImage reads the same thing to every batch
//This call is here since this is the entry point called from allocate
//Movie overwrites this function to define how it wants to load into batches
int ImagePvp::retrieveData(double timef, double dt, int batchIndex)
{
   int status = PV_SUCCESS;
   status = readPvp(inputPath, pvpFrameIdx);
   if(status != PV_SUCCESS) {
      pvLogError("%s \"%s\": retrieveData failed at t=%f with batchIndex %d\n", getKeyword(), name, timef, batchIndex);
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

int ImagePvp::readPvp(const char * filename, int frameNumber) {

   //Root process reads a PVP file into imageData.
   //All processes must be called because pvp_read_header requires that,
   //but only root process does actual I/O.
   //Scattering takes place in BaseInput::scatterInput.

   int status = PV_SUCCESS;
   int rootproc = 0;
   int rank = parent->columnId();
   Communicator * comm = parent->icCommunicator();
   PV_Stream * pvstream = PV::pvp_open_read_file(filename, comm);
   int numParams = NUM_BIN_PARAMS;
   int params[numParams];
   PV::pvp_read_header(pvstream, comm, params, &numParams);
   assert (pvstream==NULL ^ rank==rootproc); // root process needs non-null pvstream; all other processes should have null pvstream.
   if (frameNumber < 0 || frameNumber >= fileNumFrames) {
      if (rank==rootproc) {
         fprintf(stderr, "scatterImageFilePVP error: requested frameNumber %d but file \"%s\" only has frames numbered 0 through %d.\n", frameNumber, filename, params[INDEX_NBANDS]-1);
      }
      return PV_FAILURE;
   }

   if (rank!=rootproc) { return PV_SUCCESS; }

   PVLayerLoc fileloc;
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
   if (fileloc.nx != fileloc.nxGlobal || fileloc.ny != fileloc.nyGlobal ||
       nxProcs != 1 || nyProcs != 1 ||
       fileloc.kx0 != 0 || fileloc.ky0 != 0) {
       fprintf(stderr, "File \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
       exit(EXIT_FAILURE);
   }

   int bufferSize = fileloc.nx*fileloc.ny*fileloc.nf;
   if (bufferSize != imageLoc.nxGlobal*imageLoc.nyGlobal*imageLoc.nf) {
      free(imageData);
      imageData = (pvadata_t *) malloc(bufferSize*sizeof(pvadata_t));
   }
   imageLoc.nxGlobal = fileloc.nx;
   imageLoc.nyGlobal = fileloc.ny;
   imageLoc.nf = fileloc.nf;
   this->imageColorType = COLORTYPE_UNRECOGNIZED; // TODO: recognize RGB, YUV, etc.

   bool spiking = false;
   double timed = 0.0;
   int filetype = params[INDEX_FILE_TYPE];

   switch (filetype) {
   case PVP_FILE_TYPE:
      assert(0); // Is PVP_FILE_TYPE ever used?
      break;

   case PVP_ACT_FILE_TYPE:
      status = readSparseBinaryActivityFrame(numParams, params, pvstream, frameNumber);
      break;
   case PVP_ACT_SPARSEVALUES_FILE_TYPE:
      status = readSparseValuesActivityFrame(numParams, params, pvstream, frameNumber);
      break;
   case PVP_NONSPIKING_ACT_FILE_TYPE:
      status = readNonspikingActivityFrame(numParams, params, pvstream, frameNumber);
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
   pvp_close_file(pvstream, comm);

   pvpFileTime = timed;
   //This is being printed twice: track down
   //pvInfo() << "Rank " << rank << " Reading pvpFileTime " << pvpFileTime << " at timestep " << parent->simulationTime() << " with offset (" << xOffset << "," << yOffset << ")\n";
   return status;
}

int ImagePvp::readSparseBinaryActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {

   //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
   //Only need to do this once
   int status = PV_SUCCESS;
   if (needFrameSizesForSpiking) {
      pvInfo() << "Calculating file positions\n";
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
      PV::PV_fseek(pvstream, (long)8 + (long)params[INDEX_HEADER_SIZE], SEEK_SET);
      int percent = 0;
      for (int f = 0; f<params[INDEX_NBANDS]; f++) {
         int newpercent = 100*(f/(params[INDEX_NBANDS]));
         if(percent != newpercent){
            percent = newpercent;
            pvInfo() << "\r" << percent << "\% Done";
            pvInfo().flush();
         }
         //First byte position should always be 92
         if (f == 0) {
            frameStartBuf[f] = (long)92;
         }
         //Read in the number of active neurons for that frame and calculate byte position
         else {
            PV::PV_fread(&countBuf[f-1], sizeof(int), 1, pvstream);
            frameStartBuf[f] = frameStartBuf[f-1] + (long)countBuf[f-1]*(long)params[INDEX_DATA_SIZE] + (long)12;
            PV::PV_fseek(pvstream, frameStartBuf[f] - (long)4, SEEK_SET);
         }
      }
      pvInfo() << "\r" << percent << "% Done\n";
      pvInfo().flush();
      //We still need the last count
      PV::PV_fread(&countBuf[(params[INDEX_NBANDS])-1], sizeof(int), 1, pvstream);

      //So we don't have to calculate frameStart and count again
      needFrameSizesForSpiking = false;
   }

   long framepos = (long)frameStartBuf[frameNumber];
   unsigned int length = countBuf[frameNumber];
   PV::PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
   PV::PV_fread(&pvpFileTime, sizeof(double), 1, pvstream);

   unsigned int dropLength;
   PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
   assert(dropLength == length);

   memset(imageData, 0, sizeof(*imageData)*(size_t) (imageLoc.nxGlobal*imageLoc.nyGlobal*imageLoc.nf));
   int locations[length];
   PV::PV_fread(locations, sizeof(int), length, pvstream);
   for (unsigned int l=0; l<length; l++) {
      imageData[locations[l]]=(pvadata_t) 1;
   }

   return status;
}

int ImagePvp::readSparseValuesActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {

   //Allocate the byte positions in file where each frame's data starts and the number of active neurons in each frame
   //Only need to do this once
   int status = PV_SUCCESS;
   if (needFrameSizesForSpiking) {
      pvInfo() << "Calculating file positions\n";
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
      PV::PV_fseek(pvstream, (long)8 + (long)params[INDEX_HEADER_SIZE], SEEK_SET);
      int percent = 0;
      for (int f = 0; f<params[INDEX_NBANDS]; f++) {
         int newpercent = 100*(f/(params[INDEX_NBANDS]));
         if(percent != newpercent){
            percent = newpercent;
            pvInfo() << "\r" << percent << "\% Done";
            pvInfo().flush();
         }
         //First byte position should always be 92
         if (f == 0) {
            frameStartBuf[f] = (long)92;
         }
         //Read in the number of active neurons for that frame and calculate byte position
         else {
            PV::PV_fread(&countBuf[f-1], sizeof(int), 1, pvstream);
            frameStartBuf[f] = frameStartBuf[f-1] + (long)countBuf[f-1]*(long)params[INDEX_DATA_SIZE] + (long)12;
            PV::PV_fseek(pvstream, frameStartBuf[f] - (long)4, SEEK_SET);
         }
      }
      pvInfo() << "\r" << percent << "% Done\n";
      pvInfo().flush();
      //We still need the last count
      PV::PV_fread(&countBuf[(params[INDEX_NBANDS])-1], sizeof(int), 1, pvstream);

      //So we don't have to calculate frameStart and count again
      needFrameSizesForSpiking = false;
   }

   long framepos = (long)frameStartBuf[frameNumber];
   unsigned int length = countBuf[frameNumber];
   PV::PV_fseek(pvstream, framepos-sizeof(double)-sizeof(unsigned int), SEEK_SET);
   PV::PV_fread(&pvpFileTime, sizeof(double), 1, pvstream);
   {
      unsigned int dropLength;
      PV::PV_fread(&dropLength, sizeof(unsigned int), 1, pvstream);
      assert(dropLength == length);
   }
   status = PV_SUCCESS;

   memset(imageData, 0, sizeof(*imageData)*(size_t) (imageLoc.nxGlobal*imageLoc.nyGlobal*imageLoc.nf));
   struct locvalue {
      int location;
      float value;
   };
   struct locvalue locvalues[length];
   assert(sizeof(locvalue)==sizeof(int)+sizeof(float));
   PV::PV_fread(locvalues, sizeof(locvalue), length, pvstream);
   for (int l=0; l<length; l++) {
      imageData[locvalues[l].location]=(pvadata_t) locvalues[l].value;
   }
   return status;
}

int ImagePvp::readNonspikingActivityFrame(int numParams, int * params, PV_Stream * pvstream, int frameNumber) {
   int status = PV_SUCCESS;
   int recordsize = params[INDEX_RECORD_SIZE];
   int datasize = params[INDEX_DATA_SIZE];
   int framesize = recordsize*datasize+8;
   long framepos = (long)framesize * (long)frameNumber + (long)params[INDEX_HEADER_SIZE];
   //ONLY READING TIME INFO HERE
   status = PV::PV_fseek(pvstream, framepos, SEEK_SET);
   if (status != 0) {
      fprintf(stderr, "scatterImageFilePVP error: Unable to seek to start of frame %d in \"%s\": %s\n", frameNumber, pvstream->name, strerror(errno));
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      size_t numread = PV::PV_fread(&pvpFileTime, sizeof(double), (size_t) 1, pvstream);
      if (numread != (size_t) 1) {
         fprintf(stderr, "scatterImageFilePVP error: Unable to read timestamp from frame %d of file \"%s\":", frameNumber, pvstream->name);
         if (feof(pvstream->fp)) { fprintf(stderr, " end-of-file."); }
         if (ferror(pvstream->fp)) { fprintf(stderr, " fread error."); }
         fprintf(stderr, "\n");
         status = PV_FAILURE;
      }
   }
   // Assumes that imageData is of type float.
   size_t numread = PV::PV_fread(imageData, sizeof(float), (size_t) (imageLoc.nxGlobal*imageLoc.nyGlobal*imageLoc.nf), pvstream);
   return status;
}

int ImagePvp::scatterImageFilePVP(const char * filename, int xOffset, int yOffset,
                        PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber)
{

   //// Read a PVP file and scatter it to the multiple processes.
   
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
            pvInfo() << "Calculating file positions\n";
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
                  pvInfo() << "\r" << percent << "\% Done";
                  pvInfo().flush();
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
            pvInfo() << "\r" << percent << "% Done\n";
            pvInfo().flush();
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
      //pvInfo() << "Rank " << rank << " Reading pvpFileTime " << pvpFileTime << " at timestep " << parent->simulationTime() << " with offset (" << xOffset << "," << yOffset << ")\n";

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
