#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <columns/buildandrun.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include "ConvertFromTable.hpp"
#include "LocalizationProbe.hpp"
#include "MaskFromMemoryBuffer.hpp"
#define TEXTFILEBUFFERSIZE 1024

char * getImageFileName(InterColComm * icComm);
int setImageLayerMemoryBuffer(InterColComm * icComm, char const * imageFile, ImageFromMemoryBuffer * imageLayer, uint8_t ** imageBufferPtr, size_t * imageBufferSizePtr);

int main(int argc, char* argv[])
{
   int status = PV_SUCCESS;

   PV::PV_Init pv_init(&argc, &argv, true/*allowUnrecognizedArguments*/);
   // Build the column from the params file
   pv_init.registerKeyword("ConvertFromTable", createConvertFromTable);
   pv_init.registerKeyword("LocalizationProbe", createLocalizationProbe);
   pv_init.registerKeyword("MaskFromMemoryBuffer", createMaskFromMemoryBuffer);
   PV::HyPerCol * hc = pv_init.build();
   assert(hc->getStartTime()==hc->simulationTime());

   bool useHarness = true;
   for (int arg=0; arg<argc; arg++) {
      if (!strcmp(argv[arg], "--harness")) { useHarness = true; }
      if (!strcmp(argv[arg], "--no-harness")) { useHarness = false; }
   }
   if (useHarness) {
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
      LocalizationProbe * localizationProbe = NULL; 
      for (int k=0; k < hc->numberOfBaseProbes(); k++)
      {
         PV::BaseProbe * p = hc->getBaseProbe(k);
         LocalizationProbe * localization_probe = dynamic_cast<LocalizationProbe *>(p);
         if (localization_probe) {
            if (localizationProbe != NULL) {
               if (hc->columnId()==0) {
                  fprintf(stderr, "%s error: More than one LocalizationProbe (\"%s\" and \"%s\").\n",
                        argv[0], localizationProbe->getName(), localization_probe->getName());
               }
               MPI_Barrier(hc->icCommunicator()->communicator());
               exit(EXIT_FAILURE);
            }
            else {
               localizationProbe = localization_probe;
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
      if (localizationProbe==NULL) {
         if (hc->columnId()==0) {
            fprintf(stderr, "%s error: params file must have exactly one LocalizationProbe.\n",
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
         localizationProbe->setOutputFilenameBase(imageFile);
         setImageLayerMemoryBuffer(hc->icCommunicator(), imageFile, imageLayer, &imageBuffer, &imageBufferSize);
         status = hc->run(startTime, stopTime, dt);
         if (status!=PV_SUCCESS) {
            if (hc->columnId()==0) {
               fflush(stdout);
               fprintf(stderr, "Run failed at t=%f.  Exiting.\n", startTime);
            }
            break;
         }

         free(imageFile);
         imageFile = getImageFileName(hc->icCommunicator());
      }
      free(imageFile);
   }
   else {
      assert(!useHarness); // else-clause of if(useHarness) statement
      status = hc->run();
   }

   delete hc;
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
         fprintf(stdout, "Enter filename: ");
         fflush(stdout);
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
      pvError().printf("Rank %d process unable to allocate space for line from listOfImageFiles file.\n", rank);
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
   int imageNx, imageNy, imageNf;
   const uint8_t zeroVal = (uint8_t) 0;
   const uint8_t oneVal = (uint8_t) 255;
   int xStride, yStride, bandStride;
   int rank = icComm->commRank();
   if (rank==0) {
      // Doubleplusungood: much code duplication from PV::Image::readImage
      bool usingTempFile = false;
      char * path = NULL;
      if (strstr(imageFile, "://") != NULL) {
         fprintf(stdout, "Image from URL \"%s\"\n", imageFile);
         usingTempFile = true;
         std::string pathstring = "/tmp/temp.XXXXXX";
         const char * ext = strrchr(imageFile, '.');
         if (ext) { pathstring += ext; }
         path = strdup(pathstring.c_str());
         int fid;
         fid=mkstemps(path, strlen(ext));
         if (fid<0) {
            pvError().printf("Cannot create temp image file for image \"%s\".\n", imageFile);
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
                  pvError().printf("download command \"%s\" failed: %s.  Exiting\n", systemstring.c_str(), strerror(errno));
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
         fprintf(stdout, "Image from file \"%s\"\n", imageFile);
         path = strdup(imageFile);
      }
      GDALDataset * gdalDataset = PV_GDALOpen(path);
      if (gdalDataset==NULL)
      {
         pvError().printf("setImageLayerMemoryBuffer: GDALOpen failed for image \"%s\".\n", imageFile);
      }
      imageNx= gdalDataset->GetRasterXSize();
      imageNy = gdalDataset->GetRasterYSize();
      imageNf = GDALGetRasterCount(gdalDataset);
      size_t newImageBufferSize = (size_t)imageNx * (size_t)imageNy * (size_t)imageNf;
      uint8_t * imageBuffer = *imageBufferPtr;
      size_t imageBufferSize = *imageBufferSizePtr;
      if (imageBuffer==NULL || newImageBufferSize != imageBufferSize)
      {
         imageBufferSize = newImageBufferSize;
         imageBuffer = (uint8_t *) realloc(imageBuffer, imageBufferSize*sizeof(uint8_t));
         if (imageBuffer==NULL)
         {
            fprintf(stderr, "setImageLayerMemoryBuffer: Unable to create image buffer of size %d-by-%d-by-%d for image \"%s\": %s\n",
                  imageNx, imageNy, imageNf, imageFile, strerror(errno));
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
            pvError().printf("setImageLayerMemoryBuffer: Image file \"%s\", band %d, is not GDT_Byte type.\n", imageFile, iBand+1);
         }
      }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (size_t n=0; n < imageBufferSize; n++)
      {
         imageBuffer[n] = oneVal;
      }

      xStride = imageNf;
      yStride = imageNf * imageNx;
      bandStride = 1;
      gdalDataset->RasterIO(GF_Read, 0/*xOffset*/, 0/*yOffset*/, imageNx, imageNy, imageBuffer, imageNx, imageNy,
            GDT_Byte, imageNf, NULL, xStride*sizeof(uint8_t), yStride*sizeof(uint8_t), bandStride*sizeof(uint8_t));

      GDALClose(gdalDataset);
      if (usingTempFile) {
         int rmstatus = remove(path);
         if (rmstatus) {
            pvError().printf("remove(\"%s\") failed.  Exiting.\n", path);
         }    
      }
      free(path);

      *imageBufferPtr = imageBuffer;
      *imageBufferSizePtr = imageBufferSize;

      int imageDims[3];
      imageDims[0] = imageNx;
      imageDims[1] = imageNy;
      imageDims[2] = imageNf;
      MPI_Bcast(imageDims, 3, MPI_INT, 0, icComm->communicator());
   }
   else {
      int imageDims[3];
      MPI_Bcast(imageDims, 3, MPI_INT, 0, icComm->communicator());
      imageNx = imageDims[0];
      imageNy = imageDims[1];
      imageNf = imageDims[2];
      xStride = imageNf;
      yStride = imageNf * imageNx;
      bandStride = 1;
   }
   imageLayer->setMemoryBuffer(*imageBufferPtr, imageNy, imageNx, imageNf, xStride, yStride, bandStride, zeroVal, oneVal);
   return PV_SUCCESS;
}
