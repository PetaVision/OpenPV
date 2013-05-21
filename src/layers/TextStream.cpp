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
   if (getParent()->icCommunicator()->commRank()==0 && fileStream != NULL && fileStream->isfile) {
      PV_fclose(fileStream);
   }
}

int TextStream::initialize_base() {
	displayPeriod = 1;
	nextDisplayTime = 1;
	textOffset = 0;
	useCapitalization = true;
	loopInput = false;
	textBCFlag = true;
	filename = NULL;
	textData = NULL;

	return PV_SUCCESS;
}

int TextStream::initialize(const char * name, HyPerCol * hc) {
	int status = PV_SUCCESS;

	HyPerLayer::initialize(name, hc, 0);

	free(clayer->V);
	clayer->V = NULL;

	// Point to clayer data struct
    textData = clayer->activity->data;
    assert(textData!=NULL);

	// Create mpi_datatypes for border transfer
	mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

	// Exchange border information
	parent->icCommunicator()->exchange(textData, mpi_datatypes, this->getLayerLoc());

	assert(filename!=NULL);
	if( getParent()->icCommunicator()->commRank()==0 ) { // Only rank 0 should open the file pointer
		filename = strdup(filename);
		assert(filename!=NULL );

		fileStream = PV_fopen(filename, "r");
		if( fileStream->fp == NULL ) {
			fprintf(stderr, "TextStream::initialize error opening \"%s\": %s\n", filename, strerror(errno));
			status = PV_FAILURE;
			abort();
		}

		// Nav to offset if specified
		if (textOffset > 0) {
			status = PV_fseek(fileStream,textOffset,SEEK_SET);
		}
	}

	nextDisplayTime = hc->simulationTime(); //  + displayPeriod;

	status = updateState(0,parent->getDeltaTime());

	return status;
}

int TextStream::setParams(PVParams * params) {
	readUseCapitalization(params);
	readLoopInput(params);
	readTextInputPath(params);
	readDisplayPeriod(params);
	readTextOffset(params);
	readTextBCFlag(params);

	int status = HyPerLayer::setParams(params);

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
	int loc_nx = loc->nx;
	if(textBCFlag){ //Expand dimensions to the extended space
		loc_ny = loc->ny + 2*loc->nb;
		loc_nx = loc->nx + 2*loc->nb;
	}

	int numExtendedNeurons = loc_ny * loc_nx * loc->nf;

	int comm_size = comm->commSize();
	//TODO: Change to loc_ny?
	if (loc->ny % comm_size != 0) { // Need to be able to devide the number of neurons in the y (words) direction by the number of procs
		fprintf(stderr, "textStream: Number of processors must evenly devide into number of words. NumProcs=%d, NumWords=%d",comm_size,loc->ny);
		status = PV_FAILURE;
		abort();
	}

	size_t datasize = sizeof(int);
	int * temp_buffer = (int *) calloc(numExtendedNeurons, datasize);
	if (temp_buffer==NULL) {
		fprintf(stderr, "scatterTextBuffer unable to allocate memory for temp_buffer.\n");
		status = PV_FAILURE;
		abort();
	}

#ifdef PV_USE_MPI
	int rank = comm->commRank();

	if (rank==rootproc) { // Root proc should send stuff out
		for (int r=0; r<comm_size; r++) {
			readFileToBuffer(fileStream,textOffset,this->getLayerLoc(), temp_buffer);
			if (r==rootproc) {
				status = loadBufferIntoData(loc,temp_buffer);
			}
			else {
				MPI_Send(temp_buffer, numExtendedNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
			}
		}
	}
	else {
		MPI_Recv(temp_buffer, sizeof(uint4)*numExtendedNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
		status = loadBufferIntoData(loc,temp_buffer);
	}
#else // PV_USE_MPI
	readFileToBuffer(fileStream,textOffset,this->getLayerLoc(), temp_buffer);
	status = loadBufferIntoData(loc,temp_buffer);
#endif // PV_USE_MPI

	free(temp_buffer);
	temp_buffer = NULL;
	return status;
}

int TextStream::readFileToBuffer(PV_Stream * inStream, int offset, const PVLayerLoc * loc, int * buf) {
	int numReads=0;
	int numItems=1; // Number of chars to read at a time
	int encodedChar=0;
	int dataIndex=0;
	int loc_ny = loc->ny;
	int loc_nx = loc->nx;
	int y_start = 0;

	if (fileStream->filepos==0) { // Skip initial margin stuff for first read
		y_start = loc->nb;
	}

	if (textBCFlag) {
		loc_ny = loc->ny + 2*loc->nb;
		loc_nx = loc->nx + 2*loc->nb;
	}

	int preMarginReads, numExtraReads = 0;
	unsigned char * tmpChar = new unsigned char[1];  // One character at a time
	for (int y=y_start; y<loc_ny; y++) { // ny = words per proc
		while(encodedChar==0 && numReads<inStream->filelength) { // Read until nonspace
			int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
			assert(numRead==numItems);
			encodedChar = getCharEncoding(tmpChar);
			numReads += numRead;
		}
		//std::cout<<"\n---WORD---\n";
		int x=0;
		for (; x<loc_nx; x++) { // nx = num chars per word
			char charType = 'w';

			// These special characters are counted as words
			//  ! " ( ) , . : ; ? `
			if (useCapitalization) {
				if (encodedChar == 1 || encodedChar == 2 || encodedChar == 8 || encodedChar == 9 ||
						encodedChar == 12 || encodedChar == 14 || encodedChar == 26 ||
						encodedChar == 27 || encodedChar == 31 || encodedChar == 64 ||
						encodedChar == 95) {
					charType = 'p';
				}
			}
			else {
				if (encodedChar == 1 || encodedChar == 2 || encodedChar == 8 || encodedChar == 9 ||
						encodedChar == 12 || encodedChar == 14 || encodedChar == 26 ||
						encodedChar == 27 || encodedChar == 31 || encodedChar == 64 ||
						encodedChar == 69) {
					charType = 'p';
				}
			}

			if (encodedChar == 0) {
				charType = 's';
			}

			//std::cout<<"READ 1: "<<tmpChar[0]<<" is a "<<charType;

			bool break_loop = false;
			switch (charType) {
				case 'p': // Punctuation
					if (x==0) { // Punctuation is at the beginning of a word
						for (int f=0; f<loc->nf; f++) { // Store punctuation
							if (f==encodedChar) {
								buf[dataIndex] = 1;
							} else {
								buf[dataIndex] = 0;
							}
							dataIndex++;
						}
						//std::cout<<" ADDED\n";
						if (numReads<inStream->filelength) { // Read next char
							int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
							assert(numRead==numItems);
							encodedChar = getCharEncoding(tmpChar);
							numReads += numRead;
						}
					}
					break_loop = true;
					break;
				case 's': // Space
					break_loop = true;
					break;
				default: // Normal char
					for (int f=0; f<loc->nf; f++) { // Store char
						if (f==encodedChar) {
							buf[dataIndex] = 1;
						} else {
							buf[dataIndex] = 0;
						}
						dataIndex++;
					}
					//std::cout<<" ADDED\n";
					if (numReads<inStream->filelength) { // Read next char
						int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
						assert(numRead==numItems);
						encodedChar = getCharEncoding(tmpChar);
						numReads += numRead;
					}
					break_loop = false;
					break;
			}

			if (break_loop) break;
		}

		bool paddedWord = x<loc_nx;

		for (; x<loc_nx; x++) { // Fill in the rest of the word with a buffer
			for (int f=0; f<loc->nf; f++) { // Store 0
				buf[dataIndex] = 0;
				dataIndex++;
			}
		}

		while (!paddedWord && encodedChar!=0 && numReads<inStream->filelength) { // If word is longer than numCharsPerWord, read and dump the rest
			int numRead = PV_fread(tmpChar,sizeof(char),numItems,inStream);
			assert(numRead==numItems);
			encodedChar = getCharEncoding(tmpChar);
			//std::cout<<"READ 2: "<<tmpChar[0]<<" is a "<<encodedChar<<"\n";
			numReads += numRead;
		}

		if (y == loc->ny-1) {
			preMarginReads = numReads;
		}
	}

	numExtraReads = numReads - preMarginReads;
	if (textBCFlag) { // Back up to pre-margin file position
		PV_fseek(inStream,-numExtraReads,SEEK_CUR);
	}

	delete tmpChar;
	return numReads;
}

int TextStream::loadBufferIntoData(const PVLayerLoc * loc, int * buf) {
	int loc_ny = loc->ny;
	int loc_nx = loc->nx;

	if(textBCFlag){ //Expand dimensions to the extended space
		loc_ny = loc->ny + 2*loc->nb;
		loc_nx = loc->nx + 2*loc->nb;
	}

	int locIdx = 0;
	for (int y=0; y<loc_ny; y++) {          // Number of words per proc
		for (int x=0; x<loc_nx; x++) {      // Chars per word
			for (int f=0; f<loc->nf; f++) { // Char vector
				textData[locIdx] = buf[locIdx];
				locIdx += 1; // Local non-extended index
			}
		}
	}
//	locIdx = 0;
//	for (int idx=0; idx<loc_ny*loc_nx; idx++) {
//		for (int f=0; f<loc->nf; f++) {
//			if (buf[locIdx]!=0) {
//				std::cout<<f<<"  ";
//			}
//			if(textData[locIdx]!=0){
//				std::cout<<f<<"  ";
//			}
//			locIdx++;
//		}
//		std::cout<<"\n";
//	}
//	std::cout<<"\n\n\n";
//	locIdx=0;
//	for (int idx=0; idx<loc_ny*loc_nx; idx++) {
//		for (int f=0; f<loc->nf; f++) {
//			if(textData[locIdx]!=0){
//				std::cout<<f<<" ";
//			}
//			locIdx++;
//		}
//		std::cout<<"\n";
//	}
//	std::cout<<"\n\n\n";
	return PV_SUCCESS;
}

/*
 * Map input character to a integer coding set. The set includes the list of printable ASCII
 * characteres with the addition of two values for 'other' and a new line / carriage return.
 */
int TextStream::getCharEncoding(const unsigned char * printableASCIIChar) {
	int charMapValue;

	int asciiValue = (int)(unsigned char)printableASCIIChar[0];

	if (asciiValue == 10 || asciiValue == 13) { // new line or carriage return
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
				charMapValue = asciiValue - 32 - 32;
			}
		}
	}
	else {
		charMapValue = useCapitalization ? 96 : 70; // other character
	}
	if (charMapValue<0) {
		fprintf(stderr,"Char map value must be greater than or equal to 0. charMapValue = %d, asciiValue = %d, char = %s\n", charMapValue, asciiValue, printableASCIIChar);
		abort();
	}

	if (useCapitalization) {
		if (charMapValue >= 97) {
			charMapValue = 96;
		}
	}
	else {
		if (charMapValue >= 71) {
			charMapValue = 70;
		}
	}

	return charMapValue;
}


}
