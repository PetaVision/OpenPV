#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <columns/buildandrun.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include "cMakeHeader.h"
#include "PASCALCustomGroupHandler.hpp"

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

//   // These variables are only used by the root process, but must be defined here
//   // since they need to persist from one if(rank==0) statement to the next.
//   char * imageLayerName = NULL;
//   char * resultLayerName = NULL;
//   char * resultTextFile = NULL;
//   char * octaveCommand = NULL;
//   char * octaveLogFile = NULL;
//   char * classNames = NULL;
//   char * evalCategoryIndices = NULL;
//   char * displayCategoryIndices = NULL;
//   char * highlightThreshold = NULL;
//   char * heatMapThreshold = NULL;
//   char * heatMapMaximum = NULL;
//   char * drawBoundingBoxes = NULL;
//   char * boundingBoxThickness = NULL;
//   char * dbscanEps = NULL;
//   char * dbscanDensity = NULL;
//   char * heatMapMontageDir = NULL;
//   char * displayCommand = NULL;
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
//
//   BaseLayer * imageBaseLayer = hc->getLayerFromName(imageLayerName);
//   if (imageBaseLayer==NULL)
//   {
//      if (rank==0) {
//         fprintf(stderr, "%s error: no layer matches imageLayerName = \"%s\"\n", argv[0], imageLayerName);
//      }
//      status = PV_FAILURE;
//   }
//   ImageFromMemoryBuffer * imageLayer = dynamic_cast<ImageFromMemoryBuffer *>(imageBaseLayer);
//   if (imageLayer==NULL)
//   {
//      if (rank==0) {
//         fprintf(stderr, "%s error: imageLayerName = \"%s\" is not an ImageFromMemoryBuffer layer\n", argv[0], imageLayerName);
//      }
//      status = PV_FAILURE;
//   }
//
//   BaseLayer * resultBaseLayer = hc->getLayerFromName(resultLayerName);
//   if (resultBaseLayer==NULL)
//   {
//      if (rank==0) {
//         fprintf(stderr, "%s error: no layer matches resultLayerName = \"%s\"\n", argv[0], resultLayerName);
//      }
//      status = PV_FAILURE;
//   }
//   HyPerLayer * resultLayer = dynamic_cast<HyPerLayer *>(resultBaseLayer);
//   if (resultLayer==NULL)
//   {
//      if (rank==0) {
//         fprintf(stderr, "%s error: resultLayerName = \"%s\" is not a HyPerLayer\n", argv[0], resultLayerName);
//      }
//      status = PV_FAILURE;
//   }

   if (rank==0) {
      if (status != PV_SUCCESS) { exit(EXIT_FAILURE); }

//      // clobber octave logfile and result text file unless starting from a checkpoint
//      if (hc->getCheckpointReadDir()==NULL) {
//         FILE * octavefp = fopen(octaveLogFile, "w");
//         fclose(octavefp);
//         if (resultTextFile) {
//            FILE * resultTextFP = fopen(resultTextFile, "w")
//            fclose(resultTextFP);
//         }
//      }

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
//      struct stat heatMapMontageStat;
//      status = stat(heatMapMontageDir, &heatMapMontageStat);
//      if (status!=0 && errno==ENOENT) {
//         status = mkdir(heatMapMontageDir, 0770);
//         if (status!=0) {
//            fprintf(stderr, "Error: Unable to make heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
//            exit(EXIT_FAILURE);
//         }
//         status = stat(heatMapMontageDir, &heatMapMontageStat);
//      }
//      if (status!=0) {
//         fprintf(stderr, "Error: Unable to get status of heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
//         exit(EXIT_FAILURE);
//      }
//      if (!(heatMapMontageStat.st_mode & S_IFDIR)) {
//         fprintf(stderr, "Error: Heat map montage \"%s\" is not a directory\n", heatMapMontageDir);
//         exit(EXIT_FAILURE);
//      }
   }

   // Main loop: get an image, load it into the image layer, do HyPerCol::run(), lather, rinse, repeat
   char * imageFile = getImageFileName(icComm);
   while(imageFile!=NULL && imageFile[0]!='\0')
   {
      startTime = hc->simulationTime();
      stopTime = startTime + displayPeriod;
      setImageLayerMemoryBuffer(hc->icCommunicator(), imageFile, imageLayer, &imageBuffer, &imageBufferSize);
      hc->run(startTime, stopTime, dt);

//      int numParams = 20;
//      int params[numParams];
//
//      char const * imagePvpFile = rank ? 0 : imageLayer->getOutputStatePath();
//      char const * resultPvpFile = rank ? 0 : resultLayer->getOutputStatePath();
//      PV_Stream * imagePvpStream = NULL;
//      PV_Stream * resultPvpStream = NULL;
//
//      if (rank==0) {
//         imageLayer->flushOutputStateStream();
//         imagePvpStream = PV_fopen(imagePvpFile, "r", false/*verifyWrites*/);
//      }
//      status = pvp_read_header(imagePvpStream, hc->icCommunicator(), params, &numParams);
//      if (status!=PV_SUCCESS)
//      {
//         fprintf(stderr, "pvp_read_header for imageLayer \"%s\" outputfile \"%s\" failed.\n", imageLayer->getName(), imagePvpFile);
//         exit(EXIT_FAILURE);
//      }
//      if (rank==0) { PV_fclose(imagePvpStream); }
//      assert(numParams==20);
//      int imageFrameNumber = params[INDEX_NBANDS];
//
//      if (rank==0) {
//         resultLayer->flushOutputStateStream();
//         resultPvpStream = PV_fopen(resultPvpFile, "r", false/*verifyWrites*/);
//      }
//      status = pvp_read_header(resultPvpStream, hc->icCommunicator(), params, &numParams);
//      if (status!=PV_SUCCESS)
//      {
//         fprintf(stderr, "pvp_read_header for resultLayer \"%s\" outputfile \"%s\" failed.\n", resultLayer->getName(), resultPvpFile);
//         exit(EXIT_FAILURE);
//      }
//      if (rank==0) { PV_fclose(resultPvpStream); }
//      assert(numParams==20);

//      if (rank==0) {
//         int resultFrameNumber = params[INDEX_NBANDS];
//         char * basename = strrchr(imageFile, '/');
//         if (basename==NULL) { basename=imageFile; } else { basename++; }
//         basename = strdup(basename);
//         char * dot = strrchr(basename, '.');
//         if (dot) { *dot = '\0'; } // delete extension
//         std::stringstream montagePath("");
//         montagePath << heatMapMontageDir << "/" << basename << ".png";
//         free(basename);
//         std::cout << "output file is " << montagePath.str() << std::endl;
//
//         if (octavepid>0)
//         {
//            int waitstatus;
//            int waitprocess = waitpid(octavepid, &waitstatus, 0);
//            if (waitprocess < 0 && errno != ECHILD)
//            {
//               fprintf(stderr, "waitpid failed returning %d: %s (%d)\n", waitprocess, strerror(errno), errno);
//               exit(EXIT_FAILURE);
//            }
//            octavepid = 0;
//         }
//         fflush(stdout); // so that unflushed buffer isn't copied to child process
//         octavepid = fork();
//         if (octavepid < 0)
//         {
//            fprintf(stderr, "fork() error: %s\n", strerror(errno));
//            exit(EXIT_FAILURE);
//         }
//         else if (octavepid==0) {
//            /* child process */
//            std::stringstream octavecommandstream("");
//            octavecommandstream << octaveCommand <<
//                  " --eval 'load CurrentModel/ConfidenceTables/confidenceTable.mat; heatMapMontage(" <<
//                  "\"" << imagePvpFile << "\"" << ", " <<
//                  "\"" << resultPvpFile << "\"" << ", " <<
//                  "\"" << PV_DIR << "/mlab/util" << "\"" << ", " <<
//                  imageFrameNumber << ", " <<
//                  resultFrameNumber << ", " <<
//                  "confidenceTable, " <<
//                  "\"" << classNames << "\"" << ", " <<
//                  "\"" << resultTextFile << "\"" << ", " <<
//                  evalCategoryIndices << ", " <<
//                  displayCategoryIndices << ", " <<
//                  highlightThreshold << ", " <<
//                  heatMapThreshold << ", " <<
//                  heatMapMaximum << ", " <<
//                  drawBoundingBoxes << ", " <<
//                  boundingBoxThickness << ", " <<
//                  dbscanEps << ", " <<
//                  dbscanDensity << ", " <<
//                  "\"" << montagePath.str() << "\"" << ", " <<
//                  "\"" << displayCommand << "\"" <<
//                  ");'" <<
//                  " >> " << octaveLogFile << " 2>&1";
//            std::ofstream octavelogstream;
//            octavelogstream.open(octaveLogFile, std::fstream::out | std::fstream::app);
//            octavelogstream << "Calling octave with the command\n";
//            octavelogstream << octavecommandstream.str() << "\n";
//            octavelogstream.close();
//            int systemstatus = system(octavecommandstream.str().c_str()); // Analysis of the result of the current frame
//            octavelogstream.open(octaveLogFile, std::fstream::out | std::fstream::app);
//            octavelogstream << "Octave heatMapMontage command returned " << systemstatus << "\n";
//            octavelogstream.close();
//
//            exit(EXIT_SUCCESS); /* child process exits */
//         }
//         else {
//            /* parent process */
//         }
//      }
//
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
