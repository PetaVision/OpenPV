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
	TextStream(const char * name, HyPerCol * hc, const char * filename);
	virtual ~TextStream();
	virtual int updateState(double time, double dt);
	float lastUpdate()  { return lastUpdateTime; }

private:
	int initialize_base();

protected:
	TextStream();
	int initialize(const char * name, HyPerCol * hc, const char * filename);
	int getCharEncoding(const char printableASCIIChar);

	virtual void readNxScale(PVParams * params);
	virtual void readNyScale(PVParams * params);
	virtual void readNf(PVParams * params);
	virtual void readUseCapitalization(PVParams * params);
	virtual void readDisplayPeriod(PVParams * params);
	virtual void readUseTextBCFlag(PVParams * params);

	MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

	FILE * fp;

	char * filename;        // path to file if a file exists

	PVLayerLoc textLoc;     // size/location of actual image in global context
	pvdata_t * textData;    // buffer containing image

	double displayPeriod;   // Length of time a string 'frame' is displayed
	double nextDisplayTime;
	double lastUpdateTime;  // time of last image update

	bool useCapitalization; // Should mapping account for capital letters
	bool useTextBCFlag;     // Pad the end of the text with zeros

};
}

#endif /* TEXTSTREAM_HPP_ */
