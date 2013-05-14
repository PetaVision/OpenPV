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

TextStream::TextStream(const char * name, HyPerCol * hc) {
	initialize_base();
	initialize(name, hc);
}

TextStream::~TextStream() {
	filename = NULL;
	Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   if (textData != NULL) {
      delete textData;
      textData = NULL;
   }
   if (getParent()->icCommunicator()->commRank()==0 && fileStream != NULL && fileStream->isfile) {
      PV_fclose(fileStream);
   }
}

int TextStream::initialize_base() {
	displayPeriod = 1;
	nextDisplayTime = 1;
	textOffset = 0;
	useCapitalization = false;
	loopInput = false;
	filename = NULL;
	return PV_SUCCESS;
}

int TextStream::initialize(const char * name, HyPerCol * hc) {
	int status = PV_SUCCESS;

	PVParams * params = parent->parameters();
	readUseCapitalization(params);
	readLoopInput(params);

	HyPerLayer::initialize(name, hc, 0);

	free(clayer->V);
	clayer->V = NULL;

	// point to clayer data struct
    textData = clayer->activity->data;

	// create mpi_datatypes for border transfer
	mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

	// exchange border information
	parent->icCommunicator()->exchange(textData, mpi_datatypes, getLayerLoc());

	readTextInputPath(params);
	assert(filename != NULL);

	if( getParent()->icCommunicator()->commRank()==0 ) { // Only rank 0 should open the file pointer
		filename = strdup(filename);
		assert(filename != NULL );

		fileStream = PV_fopen(filename, "r");
		if( fileStream->fp == NULL ) {
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

void TextStream::readTextInputPath(PVParams * params) {
	filename = params->stringValue(name,"textInputPath",NULL);
}

void TextStream::readLoopInput(PVParams * params) {
	loopInput = (bool) params->value(name,"loopInput",loopInput);
}

void TextStream::readTextOffset(PVParams * params) {
	textOffset = params->value(name,"textOffset",textOffset);
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

      // if at end of file (EOF), exit normally or loop
      int c;
      if ((c = fgetc(fileStream->fp)) == EOF) {
    	  if (loopInput) {
			 PV_fseek(fileStream, 0L, SEEK_SET);
			 fprintf(stderr, "Text Input %s: EOF reached, rewinding file \"%s\"\n", name, filename);
    	  }
    	  else {
			  return PV_EXIT_NORMALLY;
    	  }
      }
      else {
         ungetc(c, fileStream->fp);
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
