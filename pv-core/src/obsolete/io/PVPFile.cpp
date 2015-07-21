/*
 * PVPFile.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: pschultz
 */

#include "PVPFile.hpp"
namespace PV {

PVPFile::PVPFile(const char * path, enum PVPFileMode mode, int pvpfileType, InterColComm * icComm) {
   initialize_base();
   int status = initialize(path, mode, pvpfileType, icComm);
   if (status != PV_SUCCESS) { throw errno; }
}

PVPFile::PVPFile() {
   initialize_base();
   // derived classes should call PVPFile::initialize from their initialize method, to preserve polymorphism
}

int PVPFile::initialize_base() {
   // Set member variables to safe values
   PVPFileType = 0;
   numFrames = 0;
   currentFrame = 0;
   icComm = NULL;
   stream = NULL;
   header = NULL;
   headertime = 0.0;
   return PV_SUCCESS;
}

int PVPFile::initialize(const char * path, enum PVPFileMode mode, int pvpfileType, InterColComm * icComm) {
   // For now, make sure the compiler sizes agree, but we should make the code more flexible
   assert(sizeof(int)==PVPFILE_SIZEOF_INT);
   assert(sizeof(long)==PVPFILE_SIZEOF_LONG);
   assert(sizeof(double)==PVPFILE_SIZEOF_DOUBLE);
   assert(sizeof(float)==PVPFILE_SIZEOF_FLOAT);
   assert(sizeof(short)==PVPFILE_SIZEOF_SHORT);
   
   int status = PV_SUCCESS;
   this->icComm = icComm;
   this->mode = mode;
   return initfile(path, mode, icComm);
}

int PVPFile::initfile(const char * path, enum PVPFileMode mode, InterColComm * icComm) {
   int status = PV_SUCCESS;
   const char * fopenmode = NULL;
   bool verifyWrites = false;
   errno = 0;
   switch(mode) {
   case PVPFILE_READ:
      fopenmode = "r";
      break;
   case PVPFILE_WRITE:
      fopenmode = "w";
      break;
   case PVPFILE_WRITE_READBACK:
      fopenmode = "w";
      verifyWrites = true;
      break;
   case PVPFILE_APPEND:
      fopenmode = "r+";
      struct stat statbuf;
      if (isRoot()) {
         errno = 0;
         status = PV_stat(path, &statbuf);
         if (status!=0) {
             if (errno == ENOENT) {
                // If file doesn't exist, create it and close it.
                errno = 0;
                stream = PV_fopen(path, "w", false/*verifyWrites*/);
                if (stream != NULL) {
                   status = PV_fclose(stream);
                }
             }
         }
      }
      break;
   default:
      assert(0);
      break;
   }
   if (isRoot() && status==PV_SUCCESS) {
      stream = PV_fopen(path, fopenmode, verifyWrites);
   }
   MPI_Bcast(&errno, 1, MPI_INT, rootProc(), icComm->communicator());
   status = errno ? PV_FAILURE : PV_SUCCESS;
   return status;
}

PVPFile::~PVPFile() {
   if (isRoot()) { PV_fclose(stream); }
   free(header);
}

} /* namespace PV */
