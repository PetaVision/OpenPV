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
	useTextBCFlag = true;
	return PV_SUCCESS;
}

int TextStream::initialize(const char * name, HyPerCol * hc, const char * filename) {
	int status = PV_SUCCESS;

	PVParams * params = parent->parameters();
	readUseCapitalization(params);

	HyPerLayer::initialize(name, hc, 0);

	free(clayer->V);
	clayer->V = NULL;

	// point to clayer data struct
    textData = clayer->activity->data;

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

	readDisplayPeriod(params);
	nextDisplayTime = hc->simulationTime() + displayPeriod;

	return status;
}

void TextStream::readNxScale(PVParams * params) {
   nxScale = 1; // Layer size needs to equal column size
}

void TextStream::readNyScale(PVParams * params) {
   nyScale = 1; // Layer size needs to equal column size
}

void TextStream::readNf(PVParams * params) {

	// useCapitalization  : (97) Number of printable ASCII characters + new line (\r,\n) + other
	// !useCapitalization : (71) Number of printable ASCII characters - capital letters + new line + other
    numFeatures = useCapitalization ? 95+1+1 : 95-26+1+1;
}

void TextStream::readDisplayPeriod(PVParams * params) {
	displayPeriod = params->value(name,"displayPeriod",displayPeriod);
}

void TextStream::readUseCapitalization(PVParams * params) {
	useCapitalization = (bool) params->value(name, "useCapitalization", useCapitalization);
}

void TextStream::readUseTextBCFlag(PVParams * params) {
	useTextBCFlag = (bool) params->value(name,"useTextBCFlag",useTextBCFlag);
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

      //TODO: Flag to determine if user wants to loop text file or exit normally
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
