/*
 * PVPFile.hpp
 *
 *  Created on: Jun 4, 2014
 *      Author: pschultz
 *
 * The goal of the PVPFile class is to encapsulate all interaction with
 * PVPFiles---opening and closing, reading and writing, gathering and
 * scattering---into a form where, even in the MPI context, all processes
 * call the same public methods at the same time.
 *
 * All interaction with a PVP file from outside this class should use
 * this class to work with the file.
 */

#ifndef PVPFILE_HPP_
#define PVPFILE_HPP_

#include <assert.h>
#include <sys/stat.h>
#include "../columns/InterColComm.hpp"
#include "fileio.hpp"
#include "io.h"

#define PVPFILE_SIZEOF_INT 4
#define PVPFILE_SIZEOF_LONG 8
#define PVPFILE_SIZEOF_SHORT 2
#define PVPFILE_SIZEOF_DOUBLE 8
#define PVPFILE_SIZEOF_FLOAT 4

enum PVPFileMode { PVPFILE_READ, PVPFILE_WRITE, PVPFILE_WRITE_READBACK, PVPFILE_APPEND };

namespace PV {

class PVPFile {

   // Member functions
public:
   PVPFile(const char * path, enum PVPFileMode mode, int pvpfileType, InterColComm * icComm);
   int rank() { return icComm->commRank(); }
   int rootProc() { return 0; }
   bool isRoot() { return rank()==rootProc(); }
   ~PVPFile();

protected:
   PVPFile();
   int initialize(const char * path, enum PVPFileMode mode, int pvpfileType, InterColComm * icComm);
   int initfile(const char * path, enum PVPFileMode mode, InterColComm * icComm);

private:
   int initialize_base();

   // Member variables
private:
   int PVPFileType;
   enum PVPFileMode mode;
   int numFrames;
   int currentFrame;
   InterColComm * icComm;
   
   int * header;
   double headertime;

   PV_Stream * stream;
};

} /* namespace PV */
#endif /* PVPFILE_HPP_ */
