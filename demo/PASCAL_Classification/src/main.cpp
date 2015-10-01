#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <columns/buildandrun.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include "cMakeHeader.h"
#include "PASCALCustomGroupHandler.hpp"
#include "HeatMapProbe.hpp"

char * getImageFileName(InterColComm * icComm);
int setImageLayerMemoryBuffer(InterColComm * icComm, char const * imageFile, ImageFromMemoryBuffer * imageLayer, uint8_t ** imageBufferPtr, size_t * imageBufferSizePtr);

int main(int argc, char* argv[])
{
   int status = PV_SUCCESS;

   PV::PV_Init * pv_init = new PV_Init(&argc, &argv);
   pv_init -> initialize(argc, argv);
   // Build the column from the params file
   PV::ParamGroupHandler * customGroupHandler[1];
   customGroupHandler[0] = new PASCALCustomGroupHandler();
   PV::HyPerCol * hc = build(argc, argv, pv_init, customGroupHandler, 1);
   assert(hc->getStartTime()==hc->simulationTime());

   double startTime = hc->getStartTime();
   double stopTime = hc->getStopTime();
   double dt = hc->getDeltaTime();
   double displayPeriod = stopTime - startTime;
   const int rank = hc->columnId();
   InterColComm * icComm = hc->icCommunicator();

   int layerNx, layerNy, layerNf;
   int imageNx, imageNy, imageNf;
   int bufferNx, bufferNy, bufferNf;
   size_t imageBufferSize;
   uint8_t * imageBuffer;
   int octavepid = 0; // pid of the child octave process.

   PV::ImageFromMemoryBuffer * imageLayer = NULL;
   for (int k=0; k<hc->numberOfLayers(); k++) {
      PV::HyPerLayer * l = hc->getLayer(k);
      PV::ImageFromMemoryBuffer * img_buffer_layer = dynamic_cast<PV::ImageFromMemoryBuffer *>(l);
      if (img_buffer_layer) {
         if (imageLayer!=NULL) {
            if (hc->columnId()==0) {
               fprintf(stderr, "%s error: More than one ImageFromMemoryBuffer (\"%s\" and \"%s\").\n",
                     argv[0], imageLayer->getName(), img_buffer_layer->getName());
            }
            MPI_Barrier(hc->icCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
         else {
            imageLayer = img_buffer_layer;
         }
      }
   }
   HeatMapProbe * heatMapProbe = NULL;
   for (int k=0; k < hc->numberOfBaseProbes(); k++)
   {
      PV::BaseProbe * p = hc->getBaseProbe(k);
      HeatMapProbe * heat_map_probe = dynamic_cast<HeatMapProbe *>(p);
      if (heat_map_probe) {
         if (heatMapProbe != NULL) {
            if (hc->columnId()==0) {
               fprintf(stderr, "%s error: More than one HeatMapProbe (\"%s\" and \"%s\").\n",
                     argv[0], heatMapProbe->getName(), heat_map_probe->getName());
            }
            MPI_Barrier(hc->icCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
         else {
            heatMapProbe = heat_map_probe;
         }
      }
   }
   if (imageLayer==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "%s error: params file must have exactly one ImageFromMemoryBuffer layer.\n",
               argv[0]);
         status = PV_FAILURE;
      }
   }
   if (heatMapProbe==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "%s error: params file must have exactly one HeatMapProbe.\n",
               argv[0]);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (rank==0) {
      if (status != PV_SUCCESS) { exit(EXIT_FAILURE); }

      layerNx = imageLayer->getLayerLoc()->nxGlobal;
      layerNy = imageLayer->getLayerLoc()->nyGlobal;
      layerNf = imageLayer->getLayerLoc()->nf;

      imageNx = layerNx;
      imageNy = layerNy;
      imageNf = 3;

      bufferNx = layerNx;
      bufferNy = layerNy;
      bufferNf = imageNf;

      imageBufferSize = (size_t)bufferNx*(size_t)bufferNy*(size_t)bufferNf;
      imageBuffer = NULL;
      GDALAllRegister();
   }

   // Main loop: get an image, load it into the image layer, do HyPerCol::run(), lather, rinse, repeat
   char * imageFile = getImageFileName(icComm);
   while(imageFile!=NULL && imageFile[0]!='\0')
   {
      startTime = hc->simulationTime();
      stopTime = startTime + displayPeriod;
      heatMapProbe->setOutputFilenameBase(imageFile);
      setImageLayerMemoryBuffer(hc->icCommunicator(), imageFile, imageLayer, &imageBuffer, &imageBufferSize);
      hc->run(startTime, stopTime, dt);

      free(imageFile);
      imageFile = getImageFileName(hc->icCommunicator());
   }

   delete hc;
   free(imageFile);
   delete customGroupHandler[0];
   delete pv_init;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

char * getImageFileName(InterColComm * icComm)
{
   // All processes call this routine.  Calling routine is responsible for freeing the returned string.
   char buffer[TEXTFILEBUFFERSIZE];
   int rank=icComm->commRank();
   if (rank==0)
   {
      bool found = false;
      while(!found)
      {
         printf("Enter filename: "); fflush(stdout);
         char * result = fgets(buffer, TEXTFILEBUFFERSIZE, stdin);
         if (result==NULL) { break; }

         // Ignore lines containing only whitespace
         for (char const * c = result; *c; c++)
         {
            if (!isspace(*c)) { found=true; break; }
         }
      }
      if (found)
      {
         size_t len = strlen(buffer);
         assert(len>0);
         if (buffer[len-1]=='\n') { buffer[len-1]='\0'; }
      }
      else
      {
         buffer[0] = '\0';
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(buffer, TEXTFILEBUFFERSIZE/*count*/, MPI_CHAR, 0/*rootprocess*/, icComm->communicator());
#endif // PV_USE_MPI
   char * filename = expandLeadingTilde(buffer);
   if (filename==NULL)
   {
      fprintf(stderr, "Rank %d process unable to allocate space for line from listOfImageFiles file.\n", rank);
      exit(EXIT_FAILURE);
   }
   return filename;
}

int setImageLayerMemoryBuffer(InterColComm * icComm, char const * imageFile, ImageFromMemoryBuffer * imageLayer, uint8_t ** imageBufferPtr, size_t * imageBufferSizePtr)
{
   // Under MPI, only the root process (rank==0) uses imageFile, imageBufferPtr, or imageBufferSizePtr, but nonroot processes need to call it as well,
   // because the imegeBuffer is scattered to all processes during the call to ImageFromMemoryBuffer::setMemoryBuffer().
   int layerNx = imageLayer->getLayerLoc()->nxGlobal;
   int layerNy = imageLayer->getLayerLoc()->nyGlobal;
   int layerNf = imageLayer->getLayerLoc()->nf;
   int bufferNx, bufferNy, bufferNf;
   const uint8_t zeroVal = (uint8_t) 0;
   const uint8_t oneVal = (uint8_t) 255;
   int xStride, yStride, bandStride;
   int rank = icComm->commRank();
   if (rank==0) {
      // Doubleplusungood: much code duplication from PV::Image::readImage
      bool usingTempFile = false;
      char * path = NULL;
      if (strstr(imageFile, "://") != NULL) {
         printf("Image from URL \"%s\"\n", imageFile);
         usingTempFile = true;
         std::string pathstring = "/tmp/temp.XXXXXX";
         const char * ext = strrchr(imageFile, '.');
         if (ext) { pathstring += ext; }
         path = strdup(pathstring.c_str());
         int fid;
         fid=mkstemps(path, strlen(ext));
         if (fid<0) {
            fprintf(stderr,"Cannot create temp image file for image \"%s\".\n", imageFile);
            exit(EXIT_FAILURE);
         }   
         close(fid);
         std::string systemstring;
         if (strstr(imageFile, "s3://") != NULL) {
            systemstring = "aws s3 cp \'";
            systemstring += imageFile;
            systemstring += "\' ";
            systemstring += path;
         }   
         else { // URLs other than s3://
            systemstring = "wget -O ";
            systemstring += path;
            systemstring += " \'";
            systemstring += imageFile;
            systemstring += "\'";
         }   

         int const numAttempts = MAX_FILESYSTEMCALL_TRIES;
         for(int attemptNum = 0; attemptNum < numAttempts; attemptNum++){
            int status = system(systemstring.c_str());
            if(status != 0){ 
               if(attemptNum == numAttempts - 1){ 
                  fprintf(stderr, "download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
                  exit(EXIT_FAILURE);
               }   
               else{
                  fprintf(stderr, "download command \"%s\" failed: %s.  Retrying %d out of %d.\n", systemstring.c_str(), strerror(errno), attemptNum+1, numAttempts);
                  sleep(1);
               }
            }
            else{
               break;
            }
         }
      }
      else {
         printf("Image from file \"%s\"\n", imageFile);
         path = strdup(imageFile);
      }
      GDALDataset * gdalDataset = PV_GDALOpen(path);
      if (gdalDataset==NULL)
      {
         fprintf(stderr, "setImageLayerMemoryBuffer: GDALOpen failed for image \"%s\".\n", imageFile);
         exit(EXIT_FAILURE);
      }
      int imageNx= gdalDataset->GetRasterXSize();
      int imageNy = gdalDataset->GetRasterYSize();
      int imageNf = GDALGetRasterCount(gdalDataset);
      // Need to rescale so that the the short side of the image equals the short side of the layer
      // ImageFromMemoryBuffer layer will handle the cropping.
      double xScaleFactor = (double)layerNx / (double) imageNx;
      double yScaleFactor = (double)layerNy / (double) imageNy;
      size_t imageBufferSize = *imageBufferSizePtr;
      uint8_t * imageBuffer = *imageBufferPtr;
      if (xScaleFactor < yScaleFactor) /* need to rescale so that bufferNy=layerNy and bufferNx>layerNx */
      {
         bufferNx = (int) round(imageNx * yScaleFactor);
         bufferNy = layerNy;
      }
      else {
         bufferNx = layerNx;
         bufferNy = (int) round(imageNy * xScaleFactor);
      }
      bufferNf = layerNf;
      size_t newImageBufferSize = (size_t)bufferNx * (size_t)bufferNy * (size_t)bufferNf;
      if (imageBuffer==NULL || newImageBufferSize != imageBufferSize)
      {
         imageBufferSize = newImageBufferSize;
         imageBuffer = (uint8_t *) realloc(imageBuffer, imageBufferSize*sizeof(uint8_t));
         if (imageBuffer==NULL)
         {
            fprintf(stderr, "setImageLayerMemoryBuffer: Unable to resize image buffer to %d-by-%d-by-%d for image \"%s\": %s\n",
                  bufferNx, bufferNy, bufferNf, imageFile, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }

      bool isBinary = true;
      for (int iBand=0;iBand<imageNf; iBand++)
      {
         GDALRasterBandH hBand = GDALGetRasterBand(gdalDataset,iBand+1);
         char ** metadata = GDALGetMetadata(hBand, "Image_Structure");
         if(CSLCount(metadata) > 0){
            bool found = false;
            for(int i = 0; metadata[i] != NULL; i++){
               if(strcmp(metadata[i], "NBITS=1") == 0){
                  found = true;
                  isBinary &= true;
                  break;
               }
            }
            if(!found){
               isBinary &= false;
            }
         }
         else{
            isBinary = false;
         }
         GDALDataType dataType = gdalDataset->GetRasterBand(iBand+1)->GetRasterDataType(); // Why are we using both GDALGetRasterBand and GDALDataset::GetRasterBand?
         if (dataType != GDT_Byte)
         {
            fprintf(stderr, "setImageLayerMemoryBuffer: Image file \"%s\", band %d, is not GDT_Byte type.\n", imageFile, iBand+1);
            exit(EXIT_FAILURE);
         }
      }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (size_t n=0; n < imageBufferSize; n++)
      {
         imageBuffer[n] = oneVal;
      }

      xStride = bufferNf;
      yStride = bufferNf * bufferNx;
      bandStride = 1;
      gdalDataset->RasterIO(GF_Read, 0/*xOffset*/, 0/*yOffset*/, imageNx, imageNy, imageBuffer, bufferNx, bufferNy,
            GDT_Byte, layerNf, NULL, xStride*sizeof(uint8_t), yStride*sizeof(uint8_t), bandStride*sizeof(uint8_t));

      GDALClose(gdalDataset);
      if (usingTempFile) {
         int rmstatus = remove(path);
         if (rmstatus) {
            fprintf(stderr, "remove(\"%s\") failed.  Exiting.\n", path);
            exit(EXIT_FAILURE);
         }    
      }
      free(path);

      *imageBufferPtr = imageBuffer;
      *imageBufferSizePtr = imageBufferSize;

      int buffersize[3];
      buffersize[0] = bufferNx;
      buffersize[1] = bufferNy;
      buffersize[2] = bufferNf;
      MPI_Bcast(buffersize, 3, MPI_INT, 0, icComm->communicator());
   }
   else {
      int buffersize[3];
      MPI_Bcast(buffersize, 3, MPI_INT, 0, icComm->communicator());
      bufferNx = buffersize[0];
      bufferNy = buffersize[1];
      bufferNf = buffersize[2];
      xStride = bufferNf;
      yStride = bufferNf * bufferNx;
      bandStride = 1;
   }
   imageLayer->setMemoryBuffer(*imageBufferPtr, bufferNy, bufferNx, bufferNf, xStride, yStride, bandStride, zeroVal, oneVal);
   return PV_SUCCESS;
}
