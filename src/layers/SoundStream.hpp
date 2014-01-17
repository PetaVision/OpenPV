/*
 * SoundStream.hpp
 *
 *  Created on: May 6, 2013
 *      Author: slundquist 
 */

#ifndef SOUNDSTREAM_HPP_
#define SOUNDSTREAM_HPP_

//Only compile this file and its cpp if using sound sandbox
#ifdef PV_USE_SNDFILE

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

#include <sndfile.h>

//#include "../utils/cl_random.h"

#ifndef STAT_H
#include <sys/stat.h>
#endif

namespace PV {

class SoundStream : public HyPerLayer{

public:
	SoundStream(const char * name, HyPerCol * hc);
	virtual ~SoundStream();
	virtual int allocateDataStructures();
	virtual int updateState(double time, double dt);

private:
	int initialize_base();

protected:
	SoundStream();
	int initialize(const char * name, HyPerCol * hc);

	virtual int setParams(PVParams * params);
    virtual void readSoundInputPath(PVParams * params);
	//virtual void readNxScale(PVParams * params); // Override from HyPerLayer - will just set nxScale now instead of reading
	//virtual void readNyScale(PVParams * params); // Override from HyPerLayer - will just set nyScale now instead of reading
	//virtual void readNf(PVParams * params);      // Override from HyPerLayer - will just set NF now instead of reading

	//MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange
    pvdata_t * soundData; //Buffer containing image
    SF_INFO* fileHeader;
    SNDFILE* fileStream;

	double displayPeriod;     // Length of time a string 'frame' is displayed
	double nextDisplayTime;

	const char * filename;    // Path to file if a file exists


};
}

#endif /* PV_USE_SNDFILE */

#endif /* SOUNDSTREAM_HPP_ */
