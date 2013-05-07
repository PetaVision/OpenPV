/*
 * TextStream.cpp
 *
 *  Created on: May 6, 2013
 *      Author: dpaiton
 */


#include "TextStream.hpp"

#include <stdio.h>

namespace PV {

TextStream::TextStream() {
	initialize_base();
}

TextStream::TextStream(const char * name, HyPerCol * hc, const char * filename) {
	initialize_base();
	initialize(name, hc, filename);
}

TextStream::~TextStream() {
	free(filename);
	filename = NULL;
	Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   if (textData != NULL) {
      delete textData;
      textData = NULL;
   }
   if (getParent()->icCommunicator()->commRank()==0 && fp != NULL && fp != stdout) {
      fclose(fp);
   }
}

int TextStream::initialize_base() {
	displayPeriod = 1;
	nextDisplayTime = 1;
	useCapitalization = false;
	return PV_SUCCESS;
}

int TextStream::initialize(const char * name, HyPerCol * hc, const char * filename) {
	int status = PV_SUCCESS;

	HyPerLayer::initialize(name, hc, 0);

	PVParams * params = parent->parameters();
	this->useCapitalization = (bool) params->value(name, "useCapitalization", useCapitalization);

	// nx & ny need to match HyPerCol size
	const float nxScale = 1;
	const float nyScale = 1;
    const double xScaled = -log2( (double) nxScale);
    const double yScaled = -log2( (double) nxScale);
    const int xScale = (int) nearbyint(xScaled);
    const int yScale = (int) nearbyint(yScaled);

	// useCapitalization  : (97) Number of printable ASCII characters + new line (\r,\n) + other
	// !useCapitalization : (71) Number of printable ASCII characters - capital letters + new line + other
    const int numFeatures = useCapitalization ? 95+1+1 : 95-26+1+1;

	free(clayer->V);
	clayer->V = NULL;

	// create mpi_datatypes for border transfer
	mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

	// exchange border information
	parent->icCommunicator()->exchange(textData, mpi_datatypes, getLayerLoc());

	assert(filename != NULL);

	if( getParent()->icCommunicator()->commRank()==0 ) {
		this->filename = strdup(filename);
		assert( this->filename != NULL );

		fp = fopen(filename, "r");
		if( fp == NULL ) {
			fprintf(stderr, "TextStream::initialize error opening \"%s\": %s\n", filename, strerror(errno));
			abort();
		}
	}

	displayPeriod = params->value(name,"displayPeriod", displayPeriod);
	nextDisplayTime = hc->simulationTime() + displayPeriod;

	return status;
}

int TextStream::updateState(double time, double dt)
{
	//while not eof
	//read nxGlobal words into buffer allocated to size nxGlobal
	//loop through each char
		//const char printableASCIIChar = ;
		//int charMapValue = getCharEncoding(printableASCIIChar);

      bool needNewImage = false;
      if (time >= nextDisplayTime) {
         needNewImage = true;
         nextDisplayTime += displayPeriod;
         lastUpdateTime = time;
      } // time >= nextDisplayTime

      // if at end of file (EOF), exit normally
      int c;
      if ((c = fgetc(fp)) == EOF) {
    	  return PV_EXIT_NORMALLY;
      }
      else {
         ungetc(c, fp);
      }

	// exchange border information
	parent->icCommunicator()->exchange(textData, mpi_datatypes, getLayerLoc());

   return PV_SUCCESS;
}

/*
 * Map input character to a integer coding set. The set includes the list of printable ASCII
 * characteres with the addition of two values for 'other' and a new line / carriage return.
 */
int TextStream::getCharEncoding(const char printableASCIIChar) {
	int charMapValue;

	int asciiValue = (int)printableASCIIChar;

	if (asciiValue == 11 || asciiValue == 13) {
		charMapValue = useCapitalization ? 95 : 69;
	}
	else if (asciiValue >= 32 || asciiValue <= 126) {
		if (useCapitalization) {
			charMapValue =  asciiValue - 32;
		}
		else {
			if (asciiValue < 97) {
				charMapValue = asciiValue - 32;
			}
			else {
				charMapValue = asciiValue - 26 - 32;
			}
		}
	}
	else {
		charMapValue = useCapitalization ? 96 : 70;
	}
	assert(charMapValue>=0);

	if (useCapitalization) {
		assert(charMapValue<97);
	}
	else {
		assert(charMapValue<71);
	}

	return charMapValue;
}

}
