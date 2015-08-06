#include "DisparityMovie.hpp"

namespace PV {

DisparityMovie::DisparityMovie(const char * name, HyPerCol * hc){
   DisparityMovie::initialize_base();
   Movie::initialize(name, hc);
}

int DisparityMovie::initialize_base() {
   numDisparity = 50;
   disparityIndex = 0;
   dPixelDisparity = -1;
   moveMethod = NULL;
   frameOffset = 0;
   frameCount = 0;
   return PV_SUCCESS;
}

int DisparityMovie::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = Movie::ioParamsFillGroup(ioFlag);
   ioParam_numDisparityPeriod(ioFlag);
   ioParam_dPixelDisparity(ioFlag);
   ioParam_moveMethod(ioFlag);
   return status;
}

void DisparityMovie::ioParam_numDisparityPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "numDisparity", &numDisparity, numDisparity);
}

void DisparityMovie::ioParam_dPixelDisparity(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dPixelDisparity", &dPixelDisparity, dPixelDisparity);
}

void DisparityMovie::ioParam_moveMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "moveMethod", &moveMethod, "altLeft");
   //Left eye starts first to stay still
   if(strcmp(moveMethod, "altLeft") == 0){
      frameOffset = 0;
   }
   else if(strcmp(moveMethod, "altRight") == 0){
      frameOffset = 1;
   }
}



bool DisparityMovie::updateImage(double timef, double dt){
   InterColComm * icComm = getParent()->icCommunicator();
   assert(!readPvpFile);

   if(fabs(timef - (parent->getStartTime() + parent->getDeltaTime())) > (parent->getDeltaTime()/2)){
      //If disparity is over numDisparity, read new image and reset index
      if(disparityIndex >= numDisparity - 1){
         if (filename != NULL) free(filename);
         filename = strdup(getNextFileName(skipFrameIndex));
         disparityIndex = 0;
         frameCount++;
      }
      else{
         disparityIndex++;
      }
   }
   assert(filename != NULL);

   //Set frame number (member variable in Image)
   int newOffsetX;
   if((frameCount + frameOffset) % 2 == 0){
      newOffsetX = this->offsets[0];
   }
   else{
      newOffsetX = this->offsets[0] + (disparityIndex * dPixelDisparity);
   }
   int status = readImage(filename, newOffsetX, this->offsets[1], this->offsetAnchor);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "Movie %s: Error reading file \"%s\"\n", name, filename);
      abort();
   }
   //Write to timestamp file here when updated
   if( icComm->commRank()==0 ) {
      //Only write if the parameter is set
      if(timestampFile){
         std::ostringstream outStrStream;
         outStrStream.precision(15);
         outStrStream << frameNumber << "," << timef << "," << filename << "\n";
         size_t len = outStrStream.str().length();
         int status = PV_fwrite(outStrStream.str().c_str(), sizeof(char), len, timestampFile)==len ? PV_SUCCESS : PV_FAILURE;
         if (status != PV_SUCCESS) {
            fprintf(stderr, "%s \"%s\" error: Movie::updateState failed to write to timestamp file.\n", parent->parameters()->groupKeywordFromName(name), name);
            exit(EXIT_FAILURE);
         }
         //Flush buffer
         fflush(timestampFile->fp);
      }
   }
   return true;
}



} /* namespace PV */
