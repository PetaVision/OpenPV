/*
 * RandomPatchMovie.cpp
 *
 *      Author: pschultz
 * Like Movie, but whenever the next image is loaded, offsetX and offsetY
 * are chosen randomly from the set of allowable values
 * (offset >= 0 && offsetX < imagesize - patchsize)
 */

#include "RandomPatchMovie.hpp"

namespace PV {

RandomPatchMovie::RandomPatchMovie() {
   initialize_base();
}

RandomPatchMovie::RandomPatchMovie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod) {
   initialize_base();
   initialize(name, hc, fileOfFileNames, displayPeriod);
}

int RandomPatchMovie::initialize_base() {
   numImageFiles = 0;
   imageFilenameIndices = NULL;
   listOfImageFiles = NULL;
   return PV_SUCCESS;
}

int RandomPatchMovie::initialize(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod) {
   Image::initialize(name, hc, NULL);
#ifdef PV_USE_MPI
   int rank = parent->icCommunicator()->commRank();
#else // PV_USE_MPI
   int rank = 0;
#endif // PV_USE_MPI

   int rootproc = 0;
   if( rank == rootproc) {
      FILE * fp = fopen(fileOfFileNames, "rb");
      if( fp == NULL ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to open \"%s\"\n", name, fileOfFileNames);
         fprintf(stderr, "Error code %d\n", errno);
         exit(EXIT_FAILURE);
      }
      int status = fseek(fp, 0L, SEEK_END);
      if( status != 0 ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to find end of file \"%s\"\n", name, fileOfFileNames);
         fclose(fp);
         exit(EXIT_FAILURE);
      }
      errno = 0;
      long int filelength = ftell(fp);
      if( filelength == -1L && errno ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to determine length of file \"%s\"\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      if( filelength < 0 || filelength > INT_MAX) {
         fprintf(stderr, "RandomPatchMovie \"%s\": file \"%s\" is too long.\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      fseek(fp, 0L, SEEK_SET);
      size_t filesize = (size_t) filelength;
      if( filesize == 0 ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": file of filenames \"%s\" is empty.\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      listOfImageFiles = (char *) malloc( (filesize+1) * sizeof(char) );
      if( listOfImageFiles == NULL ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to allocate memory to hold file \"%s\".\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      size_t charsread = fread(listOfImageFiles, sizeof(char), filesize, fp);
      if( charsread != filesize ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to read file \"%s\".\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      fclose(fp);
      fp = NULL;
      numImageFiles = 0;
      if( listOfImageFiles[0] == '\n' ) listOfImageFiles[0] = 0;
      for( int n=1; n<(int) filesize; n++ ) {
         if( listOfImageFiles[n] == '\n' ) {
            listOfImageFiles[n] = 0;
            if( listOfImageFiles[n-1] != 0 ) numImageFiles++;
         }
      }
      // Make sure the last file is null-terminated even if fileOfFileNames does not end with a linefeed.
      if( listOfImageFiles[filesize-1] != 0) {
         listOfImageFiles[filesize] = 0;
         numImageFiles++;
      }
      listOfImageFiles[filesize] = 0;
      if( numImageFiles == 0 ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": file \"%s\" has no non-empty lines.\n", name, fileOfFileNames);
         exit(EXIT_FAILURE);
      }
      imageFilenameIndices = (int *) malloc(numImageFiles * sizeof(int) );
      if( imageFilenameIndices == NULL ) {
         fprintf(stderr, "RandomPatchMovie \"%s\": unable to allocate memory for filename pointers.\n", name);
         exit(EXIT_FAILURE);
      }
      int fileindex = 0;
      if( listOfImageFiles[0] != 0 ){ imageFilenameIndices[0] = 0; fileindex++; }
      for( int n=1; n<(int) filesize; n++ ) {
         if( listOfImageFiles[n] != 0 && listOfImageFiles[n-1] == 0 ) {
            assert(fileindex < numImageFiles);
            imageFilenameIndices[fileindex] = n;
            fileindex++;
         }
      }
   }
   free(filename);
   filename = strdup(getRandomFilename());

   int status = getImageInfo(filename, parent->icCommunicator(), &imageLoc);
   if(status != 0) {
      fprintf(stderr, "Movie: Unable to get image info for \"%s\"\n", filename);
      exit(EXIT_FAILURE);
   }

   // create mpi_datatypes for border transfer
   PVLayerLoc * loc = &getCLayer()->loc;
   mpi_datatypes = Communicator::newDatatypes(loc);

   imageData = NULL;
   PVParams * params = hc->parameters();
   this->displayPeriod = params->value(name,"displayPeriod", displayPeriod);
   nextDisplayTime = hc->simulationTime() + this->displayPeriod;

   retrieveRandomPatch();
   readImage(filename,offsetX,offsetY);

   // exchange border information
   exchange();
   return PV_SUCCESS;
}

RandomPatchMovie::~RandomPatchMovie() {
}

int RandomPatchMovie::readOffsets() {
   // offsets are generated randomly each time an image is produced.
   offsetX = -1;
   offsetY = -1;
   return PV_SUCCESS;
}

int RandomPatchMovie::retrieveRandomPatch() {
   getImageInfo(filename, parent->icCommunicator(), &imageLoc);
   getRandomOffsets(&imageLoc, &offsetX, &offsetY);
   readImage(filename, offsetX, offsetY);

   return PV_SUCCESS;
}

int RandomPatchMovie::outputState(float timef, bool last)
{
   if (writeImages) {
      char basicFilename[PV_PATH_MAX + 1];
      snprintf(basicFilename, PV_PATH_MAX, "%s/Movie_%f.tif", parent->getOutputPath(), timef);
      write(basicFilename);
   }

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(timef);
   }

   return 0;
}

int RandomPatchMovie::updateState(float timef, float dt)
{
   updateImage(timef, dt);
   return 0;
}

bool RandomPatchMovie::updateImage(float timef, float dt) {
   bool needNewImage = timef >= nextDisplayTime;
   if( needNewImage ) {
      free(filename);
      filename = strdup(getRandomFilename());
      getImageInfo(filename, parent->icCommunicator(), &imageLoc);
      nextDisplayTime += displayPeriod;
      lastUpdateTime = timef;
      retrieveRandomPatch();
   }
   exchange();

   return true;
}

#define RANDOMPATCHMOVIE_ROOTPROC 0

char * RandomPatchMovie::getRandomFilename() {
   char * randfname;
#ifdef PV_USE_MPI
   InterColComm * icComm = parent->icCommunicator();
   int rank = icComm->commRank();
   const MPI_Comm mpi_comm = icComm->communicator();

#else // PV_USE_MPI
   int rank = 0;
#endif // PV_USE_MPI
   int rootproc = RANDOMPATCHMOVIE_ROOTPROC;
   int namelength;
   if( rank == rootproc ) {
      fileIndex = getRandomFileIndex();
      assert(fileIndex >= 0 && fileIndex < numImageFiles );
      randfname = listOfImageFiles + imageFilenameIndices[fileIndex];
   }
   namelength = (int) strlen(randfname);
#ifdef PV_USE_MPI
   MPI_Bcast(&namelength, 1, MPI_INT, rootproc, mpi_comm);
   MPI_Bcast(randfname, namelength+1, MPI_CHAR, rootproc, mpi_comm);
#endif // PV_USE_MPI
   assert( randfname != NULL );
   return randfname;
}

int RandomPatchMovie::getRandomOffsets(const PVLayerLoc * imgloc, int * offsetXptr, int * offsetYptr) {
   int xdimension = imgloc->nxGlobal - getLayerLoc()->nxGlobal;
   int ydimension = imgloc->nyGlobal - getLayerLoc()->nyGlobal;
   double x = ((double)pv_random() * xdimension)/((double) pv_random_max());
   *offsetXptr = (int) x;
   double y = ((double)pv_random() * ydimension)/((double) pv_random_max());
   *offsetYptr = (int) y;
   return PV_SUCCESS;
}

int RandomPatchMovie::getRandomFileIndex() {
#ifdef PV_USE_MPI
   assert(parent->icCommunicator()->commRank()==RANDOMPATCHMOVIE_ROOTPROC);
#endif PV_USE_MPI
   double x = ((double)pv_random() * numImageFiles)/((double) pv_random_max());
   int idx = (int) x;
   return idx;
}

}  // end namespace PV

