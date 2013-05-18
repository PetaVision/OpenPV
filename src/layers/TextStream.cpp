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
   if (textBCBuffer != NULL) {
	   delete textBCBuffer;
	   textBCBuffer=NULL;
   }
}

int TextStream::initialize_base() {
	displayPeriod = 1;
	nextDisplayTime = 1;
	textOffset = 0;
	useCapitalization = false;
	loopInput = false;
	textBCFlag = true;
	filename = NULL;
	textData = NULL;
	textBCBuffer = NULL;

	return PV_SUCCESS;
}

int TextStream::initialize(const char * name, HyPerCol * hc) {
	int status = PV_SUCCESS;

	HyPerLayer::initialize(name, hc, 0);

	PVParams * params = parent->parameters();
	setParams(params);
	numCharsPerWord = parent->getNyGlobal();

	free(clayer->V);
	clayer->V = NULL;

	// Point to clayer data struct
    textData = clayer->activity->data;
    assert(textData!=NULL);

    // Initialize text buffer
    textBCBuffer = new int[this->getLayerLoc()->nb * this->getLayerLoc()->nx * this->getLayerLoc()->nf];

	// Create mpi_datatypes for border transfer
	mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

	// Exchange border information
	parent->icCommunicator()->exchange(textData, mpi_datatypes, getLayerLoc());

	assert(filename!=NULL);
	if( getParent()->icCommunicator()->commRank()==0 ) { // Only rank 0 should open the file pointer
		filename = strdup(filename);
		assert(filename!=NULL );

		fileStream = PV_fopen(filename, "r");
		if( fileStream->fp == NULL ) {
			fprintf(stderr, "TextStream::initialize error opening \"%s\": %s\n", filename, strerror(errno));
			abort();
		}

		// Nav to offset if specified
		if (textOffset > 0) {
			status = PV_fseek(fileStream,textOffset,SEEK_SET);
		}
	}

	nextDisplayTime = hc->simulationTime() + displayPeriod;

	status = updateState(0,parent->getDeltaTime());

	return status;
}

int TextStream::setParams(PVParams * params) {
	int status = HyPerLayer::setParams(params);

	readUseCapitalization(params);
	readLoopInput(params);
	readTextInputPath(params);
	readDisplayPeriod(params);
	readTextOffset(params);
	readTextBCFlag(params);

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

void TextStream::readTextBCFlag(PVParams * params) {
	textBCFlag = params->value(name,"textBCFlag",textBCFlag);
}

int TextStream::updateState(double time, double dt)
{
	int status = PV_SUCCESS;

	bool needNewImage = false;
	if (time >= nextDisplayTime) {
		needNewImage = true;
		nextDisplayTime += displayPeriod;
		lastUpdateTime = time;
	} // time >= nextDisplayTime

	if (needNewImage) {
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

		status = scatterTextBuffer(parent->icCommunicator(),this->getLayerLoc());
	}

	return status;
}

int TextStream::scatterTextBuffer(PV::Communicator * comm, const PVLayerLoc * loc) {
	int status = PV_SUCCESS;
	int rootproc = 0;

	int loc_ny = loc->ny;
	if(textBCFlag){ //Expand dimensions to the extended space
		loc_ny = loc->ny + 2*loc->nb;
	}

	int numLocalNeurons = loc_ny * loc->nx * loc->nf;

	int comm_size = comm->commSize();
	if (loc->nx % comm_size != 0) { // Need to be able to devide the number of neurons in the x (words) direction by the number of procs
		fprintf(stderr, "textStream: Number of processors must evenly devide into number of words");
		status = PV_FAILURE;
		abort();
	}

#ifdef PV_USE_MPI
	int rank = comm->commRank();

	size_t datasize = sizeof(int);
	int * temp_buffer = (int *) calloc(numLocalNeurons, datasize);
	if (temp_buffer==NULL) {
		fprintf(stderr, "scatterActivity unable to allocate memory for temp_buffer.\n");
		status = PV_FAILURE;
		abort();
	}

	if (rank==rootproc) { // Root proc should send stuff out
		readFileToBuffer(fileStream,textOffset,this->getLayerLoc(), temp_buffer);
		for (int r=0; r<comm_size; r++) {
			if (r==rootproc) {
				status = loadBufferIntoData(loc,temp_buffer);
			}
			else {
				MPI_Send(temp_buffer, numLocalNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
			}
		}

	}
	else {
		MPI_Recv(temp_buffer, sizeof(uint4)*numLocalNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
		status = loadBufferIntoData(loc,temp_buffer);
	}
#else // PV_USE_MPI
	readFileToBuffer(fileStream,textOffset,this->getLayerLoc(), tmpFileBuf);
	status = loadBufferIntoData(loc,temp_buffer);
#endif // PV_USE_MPI

	free(temp_buffer);
	return status;
}

int TextStream::readFileToBuffer(PV_Stream * inStream, int offset, const PVLayerLoc * loc, int * buf) {
	int numReads = 0;
	int numItems = 1; // Number of chars to read at a time
	int encodedChar;
	int dataIndex = 0;
	int loc_ny = loc->ny;
	int yEnd = loc_ny;

	if (textBCFlag) {
		loc_ny = loc->ny + 2*loc->nb;
		yEnd = loc_ny - loc->nb;

		dataIndex = loc->nb*loc->nx*loc->nf;
		if (inStream->filepos != 0) { // Not at beginning of file
			for (int idx=0; idx<dataIndex; idx++) {
				buf[idx] = textBCBuffer[idx];
			}
		}
	}

	char * tmpChar = new char[1];  // One character at a time
	for (int y=0; y<yEnd; y++) {  // ny = words per proc
		encodedChar = NAN;
		bool punctChar = false; // Set if punctuation was read
		for (int x=0; x<loc->nx; x++) { // nx = numCharsPerWord
			// Only read from file if previous char was not a space
			// Also, only read if not at the end of the file
			// Also, only read if last char was not punctuation
			if (encodedChar!=0 && numReads<inStream->filelength && !punctChar) {
				int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
				assert(numRead==numItems);
				encodedChar = getCharEncoding(tmpChar);
				numReads += numRead;
			}

			// These special characters are counted as words
			//  ! " ( ) , . : ; ? `
			if (encodedChar == 1 || encodedChar == 2 || encodedChar == 8 || encodedChar == 9 ||
					encodedChar == 12 || encodedChar == 14 || encodedChar == 26 ||
					encodedChar == 27 || encodedChar == 31 || encodedChar == 64) {
				punctChar = true;
			}

			for (int f=0; f<loc->nf; f++) { //nf = numFeatures (num printable ascii chars w/ or w/out caps)
				if (punctChar) {
					if (x==0) { // If punctuation the start of the word
						if (f==encodedChar) {
							buf[dataIndex] = 1;
						} else {
							buf[dataIndex] = 0;
						}
					} else {
						buf[dataIndex] = 0;
					}
				} else {
					if (f==encodedChar) {
						buf[dataIndex] = 1;
					} else {
						buf[dataIndex] = 0;
					}
				}
				dataIndex++;
			}
		}

		if (punctChar) { // If you had read a punctuation mark, back out the file pointer
			PV_fseek(inStream,-1,SEEK_CUR); // back up FP to re-read
		}

		while (encodedChar !=0 && numReads<inStream->filelength) { // If word is longer than numCharsPerWord, read and dump the rest
			int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
			assert(numRead==numItems);
			encodedChar = getCharEncoding(tmpChar);
			numReads += numRead;
		}
	}

	if (textBCFlag) {
		dataIndex = 0;
		long desiredFP = inStream->filepos;
		encodedChar = NAN;
		for (int b=0; b<loc->nb; b++) {
			for (int x=0; x<loc->nx; x++) {
				for (int f=0; f<loc->nf; f++) {
					if (b <= loc->nb) { // Fill in the end of the last thing read
						textBCBuffer[dataIndex] = buf[loc_ny*loc->nx*loc->nf - (loc->nb*loc->nx*loc->nf) + dataIndex];
					}
					dataIndex += 1;
				}
			}
		}
		PV_fseek(fileStream,-(desiredFP-inStream->filepos),SEEK_CUR); // Move back to where you were
	}

	delete tmpChar;
	return numReads;
}

int TextStream::loadBufferIntoData(const PVLayerLoc * loc, int * buf) {
	int loc_ny = loc->ny;
	if(textBCFlag){ //Expand dimensions to the extended space
		loc_ny = loc->ny + 2*loc->nb;
	}

	int locIdx = 0;
	for (int y=0; y<loc_ny; y++) {         // number of words per proc
		for (int x=0; x<loc->nx; x++) {     // Chars per word
			for (int f=0; f<loc->nf; f++) { // Char vector
				int extLocIdx = kIndexExtended(locIdx,loc->nx,loc_ny,loc->nf,loc->nb);
				textData[extLocIdx] = buf[locIdx];
				locIdx += 1; // Local non-extended index
			}
		}
	}
	return PV_SUCCESS;
}

/*
 * Map input character to a integer coding set. The set includes the list of printable ASCII
 * characteres with the addition of two values for 'other' and a new line / carriage return.
 */
int TextStream::getCharEncoding(const char * printableASCIIChar) {
	int charMapValue;

	int asciiValue = (int)printableASCIIChar[0];

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
