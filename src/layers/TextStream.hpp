/*
 * TextStream.hpp
 *
 *  Created on: May 6, 2013
 *      Author: dpaiton
 */

#ifndef TEXTSTREAM_HPP_
#define TEXTSTREAM_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include "../utils/cl_random.h"

namespace PV {

class TextStream : public HyPerLayer{

public:
	TextStream(const char * name, HyPerCol * hc);
	virtual ~TextStream();
	virtual int updateState(double time, double dt);
	float lastUpdate()  { return lastUpdateTime; }

private:
	int initialize_base();

protected:
	TextStream();
	int initialize(const char * name, HyPerCol * hc);
	int getCharEncoding(const char printableASCIIChar);

	virtual void readNxScale(PVParams * params); // Override from HyPerLayer - will just set nxScale now instead of reading
	virtual void readNyScale(PVParams * params); // Override from HyPerLayer - will just set nyScale now instead of reading
	virtual void readNf(PVParams * params);      // Override from HyPerLayer - will just set NF now instead of reading
	virtual void readUseCapitalization(PVParams * params);
	virtual void readLoopInput(PVParams * params);
	virtual void readDisplayPeriod(PVParams * params);
	virtual void readTextInputPath(PVParams * params);
	virtual void readTextOffset(PVParams * params);

	int scatterTextFile(const char * filename, int xOffset, int yOffset,
                         PV::Communicator * comm, const PVLayerLoc * loc, float * buf);

	MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

	PV_Stream * fileStream;

	const char * filename;        // Path to file if a file exists

	PVLayerLoc textLoc;     // Size/location of actual image in global context
	pvdata_t * textData;    // Buffer containing image

	double displayPeriod;   // Length of time a string 'frame' is displayed
	double nextDisplayTime;
	double lastUpdateTime;  // Time of last image update

	int textOffset;         // Starting point for run

	bool useCapitalization; // Should mapping account for capital letters
	bool loopInput;         // Should the algorithm loop through the text file until specified total run time is completed or exit gracefully

};
}

#endif /* TEXTSTREAM_HPP_ */
