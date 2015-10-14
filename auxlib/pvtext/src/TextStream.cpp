/*
 * TextStream.cpp
 *
 *  Created on: May 6, 2013
 *      Author: dpaiton
 */


#include "TextStream.hpp"

#include <stdio.h>

namespace PVtext {

TextStream::TextStream() {
   initialize_base();
}

TextStream::TextStream(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

TextStream::~TextStream() {
   filename = NULL;
   PV::Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
   if (getParent()->icCommunicator()->commRank()==0 && fileStream != NULL && fileStream->isfile) {
      PV::PV_fclose(fileStream);
   }
}

int TextStream::initialize_base() {
   numChannels = 0;
   displayPeriod = 1;
   //nextDisplayTime = 1;
   textOffset = 0;
   useCapitalization = true;
   loopInput = false;
   textBCFlag = true;
   encodedChar = 0;
   filename = NULL;
   textData = NULL;

   return PV_SUCCESS;
}

int TextStream::initialize(const char * name, PV::HyPerCol * hc) {
   int status = PV_SUCCESS;

   HyPerLayer::initialize(name, hc);

   assert(parent->numCommColumns()==1); // Can't split up by letters, only by words (rows, or y)

   assert(filename!=NULL);
   if( getParent()->icCommunicator()->commRank()==0 ) { // Only rank 0 should open the file pointer
      filename = strdup(filename);
      assert(filename!=NULL );

      fileStream = PV::PV_fopen(filename, "r", false/*verifyWrites*/);
      if( fileStream->fp == NULL ) {
         fprintf(stderr, "TextStream::initialize error opening \"%s\": %s\n", filename, strerror(errno));
         status = PV_FAILURE;
         abort();
      }

      // Nav to offset if specified
      if (textOffset > 0) {
         status = PV::PV_fseek(fileStream,textOffset,SEEK_SET);
      }
   }

   return status;
}

int TextStream::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   PV::PVParams * params = parent->parameters();
   ioParam_useCapitalization(ioFlag); // This line needs to be before HyPerLayer::ioParamsFillGroup call, since numFeatures depends on useCapitalization
   ioParam_loopInput(ioFlag);
   ioParam_textInputPath(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_textOffset(ioFlag);
   ioParam_textBCFlag(ioFlag);

   int status = HyPerLayer::ioParamsFillGroup(ioFlag);

   return status;
}

void TextStream::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) nxScale = 1; // Layer size needs to equal column size
}

void TextStream::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) nyScale = 1; // Layer size needs to equal column size
}

void TextStream::ioParam_nf(enum ParamsIOFlag ioFlag) {
   // useCapitalization  : (97) Number of printable ASCII characters + new line (\r,\n) + other
   // !useCapitalization : (71) Number of printable ASCII characters - capital letters + new line + other
   if (ioFlag == PARAMS_IO_READ) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "useCapitalization "));
      numFeatures = useCapitalization ? 95+1+1 : 95-26+1+1;
   }
}

void TextStream::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, displayPeriod);
}

void TextStream::ioParam_useCapitalization(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "useCapitalization", &useCapitalization, useCapitalization);
}

void TextStream::ioParam_textInputPath(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "textInputPath", &filename, "random");
   // filename = params->stringValue(name,"textInputPath","random");
   // Presumably "random" was meant as a default value when passing it to PVParams::stringValue
   // but PVParams::stringValue doesn't have default values (the third argument is bool warnIfAbsent).
}

void TextStream::ioParam_loopInput(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "loopInput", &loopInput, loopInput);
}

void TextStream::ioParam_textOffset(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "textOffset", &textOffset, textOffset);
}

void TextStream::ioParam_textBCFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "textBCFlag", &textBCFlag, textBCFlag);
}

void TextStream::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      mirrorBCflag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "mirrorBCflag", false);
   }
} // Flag doesn't make sense for text

void TextStream::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   return;
}

int TextStream::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   // Point to clayer data struct
   textData = clayer->activity->data;
   assert(textData!=NULL);

   // Create mpi_datatypes for border transfer
   mpi_datatypes = PV::Communicator::newDatatypes(getLayerLoc());
   
   return status;
}

int TextStream::setActivity() {
   // Exchange border information
   parent->icCommunicator()->exchange(textData, mpi_datatypes, this->getLayerLoc());

   int status = updateState(0,parent->getDeltaTime());

   return status;
}

int TextStream::allocateV() {
   free(clayer->V);
   clayer->V = NULL;
   return PV_SUCCESS;
}

double TextStream::getDeltaUpdateTime(){
   return displayPeriod;
}

//bool TextStream::needUpdate(double time, double dt){
//   if (time >= nextDisplayTime) {
//      nextDisplayTime += displayPeriod*dt;
//      return true;
//   } // time >= nextDisplayTime
//   else{
//      return false;
//   }
//}

int TextStream::updateState(double time, double dt)
{
   int status = PV_SUCCESS;

   int rootproc = 0;

   ////Moved to needUpdate function
   //bool needNewText = false;
   //if (time >= nextDisplayTime) {
   //   needNewText = true;
   //   nextDisplayTime += displayPeriod*dt;
   //   lastUpdateTime = time;
   //} // time >= nextDisplayTime
   //
   //if (needNewText) {
   if (parent->columnId() == rootproc) {
      // if at end of file (EOF), exit normally or loop
      int c = fgetc(fileStream->fp);
      if (c == EOF) {
         if (loopInput) {
            PV::PV_fseek(fileStream, 0L, SEEK_SET);
            fprintf(stderr, "Text Input %s: EOF reached, rewinding file \"%s\".\n", name, filename);
         }
         else {
            fprintf(stderr, "Text Input %s: EOF reached, exiting normally from file \"%s\".\n", name, filename);
            return PV_EXIT_NORMALLY;
         }
      }
      else {
         ungetc(c, fileStream->fp);
      }
   } // (parent->columnId() == rootproc)

   status = scatterTextBuffer(parent->icCommunicator(),this->getLayerLoc());
   //}

   return status;
}

int TextStream::scatterTextBuffer(PV::Communicator * comm, const PVLayerLoc * loc) {
   int status = PV_SUCCESS;
   int rootproc = 0;

   int loc_ny = loc->ny;
   int loc_nx = loc->nx;
   if(textBCFlag){ //Expand dimensions to the extended space
      loc_ny = loc->ny + loc->halo.dn + loc->halo.up;
      loc_nx = loc->nx + loc->halo.lt + loc->halo.rt;
   }

   int numExtendedNeurons = loc_ny * loc_nx * loc->nf;

   //   //TODO: Change to loc_ny?
   //   if (loc->ny % comm_size != 0) { // Need to be able to devide the number of neurons in the y (words) direction by the number of procs
   //      fprintf(stderr, "textStream: Number of processors must evenly devide into number of words. NumProcs=%d, NumWords=%d",comm_size,loc->ny);
   //      status = PV_FAILURE;
   //      abort();
   //   }

   //TODO: Would it be more efficient to move this to initialize?
   size_t datasize = sizeof(int);
   int * temp_buffer = (int *) calloc(numExtendedNeurons, datasize); // This buffer is the size of the given rank's activity buffer
   if (temp_buffer==NULL) {
      fprintf(stderr, "textStream: scatterTextBuffer unable to allocate memory for temp_buffer.\n");
      status = PV_FAILURE;
      abort();
   }

#ifdef PV_USE_MPI
   int comm_size = comm->commSize();
   int rank = comm->commRank();

   if (rank==rootproc) { // Root proc should send stuff out
      for (int r=0; r<comm_size; r++) {
         //TODO: Why do I need to do this? readFileToBuffer should fill buffer completely with values and overwrite all of the buffer
         for (int buffIdx=0; buffIdx < numExtendedNeurons; buffIdx++) {
            temp_buffer[buffIdx] = 0;
         }
         //Reading different buffers because file position changes
         status = readFileToBuffer(textOffset,this->getLayerLoc(), temp_buffer);
         if (r==rootproc) {
            status = loadBufferIntoData(loc,temp_buffer);
         }
         else {
            MPI_Send(temp_buffer, numExtendedNeurons*(int) datasize, MPI_BYTE, r, 171+r/*tag*/, comm->communicator());
         }
         if (fileStream->filepos >= fileStream->filelength) {
            if (loopInput) {
               PV::PV_fseek(fileStream, 0L, SEEK_SET);
               fprintf(stderr, "Text Input %s: EOF reached, rewinding file \"%s\".\n", name, filename);
            }
            else {
               fprintf(stderr, "Text Input %s: EOF reached, exiting normally from file \"%s\".\n", name, filename);
               return PV_EXIT_NORMALLY;
            }
         }
      }
   }
   else {
      MPI_Recv(temp_buffer, sizeof(uint4)*numExtendedNeurons, MPI_BYTE, rootproc, 171+rank/*tag*/, comm->communicator(), MPI_STATUS_IGNORE);
      status = loadBufferIntoData(loc,temp_buffer);
   }
#else // PV_USE_MPI
   status = readFileToBuffer(textOffset,this->getLayerLoc(), temp_buffer);
   status = loadBufferIntoData(loc,temp_buffer);
   if (totRead >= fileStream->filelength) {
      if (loopInput) {
         PV::PV_fseek(fileStream, 0L, SEEK_SET);
         fprintf(stderr, "Text Input %s: EOF reached, rewinding file \"%s\".\n", name, filename);
      }
      else {
         fprintf(stderr, "Text Input %s: EOF reached, exiting normally from file \"%s\".\n", name, filename);
         return PV_EXIT_NORMALLY;
      }
   }
#endif // PV_USE_MPI

   free(temp_buffer);
   temp_buffer = NULL;
   return status;
}

char TextStream::getCharType(int encodedChar){
   char charType = 'l'; // Initialize to letter

   // These special characters are counted as words
   //  ! " ( ) , . : ; ? `
   if (useCapitalization) {
      if (encodedChar==1 || encodedChar==2 || encodedChar==8 || encodedChar==9 ||
            encodedChar==12 || encodedChar==13 || encodedChar==14 ||
            encodedChar==26 || encodedChar==27 || encodedChar==28 ||
            encodedChar==31 || encodedChar==32 || encodedChar==64 ||
            encodedChar==95) {
         charType = 'p';
      }
   }
   else {
      if (encodedChar==1 || encodedChar==2 || encodedChar==8 || encodedChar==9 ||
            encodedChar==12 || encodedChar==13 || encodedChar==14 ||
            encodedChar==26 || encodedChar==27 || encodedChar==28 ||
            encodedChar==31 || encodedChar== 32 || encodedChar==64 ||
            encodedChar==69) {
         charType = 'p';
      }
   }
   if (encodedChar == 0) {
      charType = 's';
   }
   return charType;
}

int TextStream::readFileToBuffer(int offset, const PVLayerLoc * loc, int * buf) {
   int status = PV_SUCCESS;
   int numItems=1; // Number of chars to read at a time
   int loc_ny = loc->ny;
   int loc_nx = loc->nx;
   int y_start = 0;
   int x_start = 0;

   if (textBCFlag) {
      loc_ny = loc->ny + loc->halo.lt + loc->halo.rt;
      loc_nx = loc->nx + loc->halo.lt + loc->halo.rt;
      x_start = loc->halo.lt;
   }

   unsigned char * tmpChar = new unsigned char[1];  // One character at a time
   char charType;
   if (fileStream->filepos==0) { // Skip initial margin stuff for first read
      y_start = loc->halo.up;
   }

   int numCharReads=0, preMarginReads=0, numExtraReads = 0;
   for (int y=y_start; y<loc_ny; y++) { // ny = words per proc
      //std::cout<<"EDCHR: "<<encodedChar<<"\n";
      //Remove all extra whitespaces before word
      while(encodedChar==0 && fileStream->filepos + numItems <= fileStream->filelength) { // Read until nonspace
         int numRead = PV::PV_fread(tmpChar,sizeof(char),numItems,fileStream);
         assert(numRead==numItems);
         numCharReads += numRead;
         encodedChar = getCharEncoding(tmpChar);
         //std::cout<<"LKSPC: "<<tmpChar[0]<<" encoded as "<<encodedChar<<"\n";
      }
      if (fileStream->filepos >= fileStream->filelength) {
         return status;
      }
      //std::cout<<"\n---WORD---\n";
      int x=x_start;
      //std::cout << "loc->nx" << loc->nx << " loc->nb " << loc->nb << "\n";
      //Read until buffer + one side of margin
      //x_start is loc->nb
      for (; x<loc->nx+loc->halo.lt; x++) { // nx = num chars per word
         charType = getCharType(encodedChar);
         //std::cout<<"READ 1: "<<tmpChar[0]<<" is a "<<charType;
         bool break_loop = false;
         switch (charType) {
         case 'p': // Punctuation
            if (x==x_start) { // Punctuation is at the beginning of a word
               for (int f=0; f<loc->nf; f++) { // Store punctuation
                  if (f==encodedChar) {
                     buf[loc->nf*(loc_nx*y+x)+f] = 1;
                  } else {
                     buf[loc->nf*(loc_nx*y+x)+f] = 0;
                  }
               }
               //std::cout<<" ADDED Punct "<<encodedChar<<"; x="<<x<<"\n";
               if (fileStream->filepos + numItems <= fileStream->filelength) { // Read next char
                  int numRead = PV::PV_fread(tmpChar,sizeof(char),numItems,fileStream);
                  assert(numRead==numItems);
                  numCharReads += numRead;
                  encodedChar = getCharEncoding(tmpChar);
                  //std::cout<<"PUTRED: "<<tmpChar[0]<<" is a "<<charType<<"encoded as "<<encodedChar<<"\n";
               } else {
                  return status;
               }
            }
            x++; // Need to increment X so padding does not take out char
            break_loop = true;
            break;
         case 's': // Space
            break_loop = true;
            break;
         default: // Normal char (letter)
            for (int f=0; f<loc->nf; f++) { // Store char
               if (f==encodedChar) {
                  buf[loc->nf*(loc_nx*y+x)+f] = 1;
               } else {
                  buf[loc->nf*(loc_nx*y+x)+f] = 0;
               }
            }
            //std::cout<<" ADDED letter "<<encodedChar<<"; x="<<x<<"\n";
            //Word too long
            if (x==loc->nx+loc->halo.lt-1) {
               char tempCharType = getCharType(encodedChar);
               //Look for spaces, return, puncuation
               while(encodedChar!=0 && tempCharType!='p' && fileStream->filepos + numItems <= fileStream->filelength) { // Dump the rest of the word
                  int numRead = PV::PV_fread(tmpChar,sizeof(char),numItems,fileStream);
                  assert(numRead==numItems);
                  numCharReads += numRead;
                  encodedChar = getCharEncoding(tmpChar);
                  //std::cout<<"DELET: "<<tmpChar[0]<<" encoded as "<<encodedChar<<"\n";
               }
               x++; // Increment X if breaking loop & a char has been added
               break_loop = true;
            }
            //Continue with word
            else {
               if (fileStream->filepos + numItems <= fileStream->filelength) {
                  // Read next char
                  int numRead = PV::PV_fread(tmpChar,sizeof(char),numItems,fileStream);
                  assert(numRead==numItems);
                  numCharReads += numRead;
                  encodedChar = getCharEncoding(tmpChar);
               } else {
                  return status;
               }
               break_loop = false;
            }
            break;
         }//End switch statement

         if (break_loop) break;
      }//End reading characters

      for (; x<loc->nx+loc->halo.lt; x++) { // Fill in the rest of the word with a buffer
         for (int f=0; f<loc->nf; f++) { // Store 0
            buf[loc->nf*(loc_nx*y+x)+f] = 0;
         }
      }

      //When it's gotten to the end of the restricted layer
      //Saving until loc->ny since that's where the beginning margin of next proc is
      if (y == loc->ny - 1) {
         preMarginReads = numCharReads;
      }
   }

   numExtraReads = numCharReads - preMarginReads;
   if (textBCFlag) { // Back up to pre-margin file position
      PV::PV_fseek(fileStream,-1 * numExtraReads,SEEK_CUR);
   }

   delete[] tmpChar;
   return status;
}

int TextStream::loadBufferIntoData(const PVLayerLoc * loc, int * buf) {
   int loc_ny = loc->ny;
   int loc_nx = loc->nx;

   if(textBCFlag){ //Expand dimensions to the extended space
      loc_ny = loc->ny + loc->halo.dn + loc->halo.up;
      loc_nx = loc->nx + loc->halo.dn + loc->halo.up;
   }

   //TODO: Get memcpy to work
   // memcpy(buf, textData, loc_ny*loc_nx*loc->nf*sizeof(pvdata_t));

   for (int y=0; y<loc_ny; y++) {          // Number of words per proc
      for (int x=0; x<loc_nx; x++) {      // Chars per word
         for (int f=0; f<loc->nf; f++) { // Char vector
            textData[loc->nf*(loc_nx*y+x)+f] = buf[loc->nf*(loc_nx*y+x)+f];
         }
      }
   }

   //   std::cout<<"----- RANK = "<<parent->columnId()<<"-----\n";
   //   locIdx = 0;
   //   for (int idx=0; idx<loc_ny*loc_nx; idx++) {
   //      for (int f=0; f<loc->nf; f++) {
   //         if (buf[locIdx]!=0) {
   //            std::cout<<f<<"  ";
   //         }
   //         if(textData[locIdx]!=0){
   //            std::cout<<f<<"  ";
   //         }
   //         locIdx++;
   //      }
   //      std::cout<<"\n";
   //   }
   //   std::cout<<"\n\n\n";

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
      charMapValue = useCapitalization ? 96 : 70; // other character
   }
   if (useCapitalization) {
      if (charMapValue > 96) {
         charMapValue = 96;
      }
   }
   else {
      if (charMapValue > 70) {
         charMapValue = 70;
      }
   }

   return charMapValue;
}


}
