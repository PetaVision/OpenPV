/*
 * TextStreamProbe.cpp
 *
 *  Created on: May 20, 2013
 *      Author: pschultz
 */

#include "TextStreamProbe.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

TextStreamProbe::TextStreamProbe() {
   initTextStreamProbe_base();
}

TextStreamProbe::TextStreamProbe(const char * filename, HyPerLayer * layer, pvdata_t display_period) {
   initTextStreamProbe_base();
   initTextStreamProbe(filename, layer, display_period);
}

TextStreamProbe::~TextStreamProbe() {
}

int TextStreamProbe::initTextStreamProbe_base() {
   return PV_SUCCESS;
}

int TextStreamProbe::initTextStreamProbe(const char * filename, HyPerLayer * layer, pvdata_t display_period) {
   int status = LayerProbe::initLayerProbe(filename, layer);
   displayPeriod = display_period;
   nextDisplayTime = 0.0f;
   int nf = layer->getLayerLoc()->nf;
   switch(nf) {
   case 97:
      useCapitalization = true;
      break;
   case 71:
      useCapitalization = false;
      break;
   default:
      fprintf(stderr, "TextStreamProbe error: layer \"%s\" must have either 97 or 71 features.\n", getTargetLayer()->getName());
      break;
   }
   return status;
}

int TextStreamProbe::outputState(double timef) {
   if (timef<nextDisplayTime) return PV_SUCCESS;
   nextDisplayTime += displayPeriod;
   int status = PV_SUCCESS;
   assert(getTargetLayer()->getParent()->icCommunicator()->numCommColumns()==1);
   int num_rows = getTargetLayer()->getParent()->icCommunicator()->numCommRows();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int nx = loc->nx;
   assert(nx==loc->nxGlobal);
   int ny = loc->ny;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;
   int nb = loc->nb;
   assert(nyGlobal==ny*num_rows);
   MPI_Comm mpi_comm = getTargetLayer()->getParent()->icCommunicator()->communicator();
   pvdata_t * buf = (pvdata_t *) calloc(ny*nf*nx, sizeof(pvdata_t)); // Buffer holding the max feature value;

   int rootproc = 0;
   if (getTargetLayer()->getParent()->columnId()==rootproc) {
      char * cbuf = (char *) calloc(2*nx, sizeof(char)); // Translation of feature numbers into characters.  2x because nonprintable characters
      for (int proc=0; proc<num_rows; proc++) {
         if (proc==rootproc) {
            // Copy to layer data to buf.
            for (int y=0; y<ny; y++) {
               int kex = kIndexExtended(y*nx*nf, nx, ny, nf, nb);
               memcpy(&buf[y*nx*nf], &getTargetLayer()->getLayerData()[kex], nx*nf*sizeof(pvdata_t));
            }
         }
         else {
            MPI_Recv(buf, ny*nx*nf, MPI_FLOAT, proc, 157, mpi_comm, MPI_STATUS_IGNORE);
         }
         for (int y=0; y<ny; y++) {
            char * curcbuf = cbuf;
            for (int x=0; x<nx; x++) {
               pvdata_t fmax = -FLT_MAX;
               int floc = -1;
               for (int f=0; f<nf; f++) {
                  if (buf[nf*(nx*y+x)+f]>fmax) {
                     fmax=buf[nf*(nx*y+x)+f];
                     floc = f;
                  }
               }
               assert(floc>=0 && floc < nf);
               // Now floc is the location of the maximum over f, and fmax is the value.
               featureNumberToCharacter(floc, &curcbuf, cbuf, 2*nx*ny);
            }
            assert(curcbuf-cbuf<2*nx*ny);
            *curcbuf = '\0';
            fprintf(outputstream->fp, "%s ", cbuf);
         }
      }
      //fprintf(outputstream->fp, "\n");
      free(cbuf); cbuf = NULL;
   }
   else {
      for (int y=0; y<ny; y++) {
         int kex = kIndexExtended(y*nx*nf, nx, ny, nf, nb);
         memcpy(&buf[y*nx*nf], &getTargetLayer()->getLayerData()[kex], nx*nf*sizeof(pvdata_t));
      }
      MPI_Send(buf, ny*nx*nf, MPI_FLOAT, rootproc, 157, mpi_comm);
   }
   free(buf); buf=NULL;

   return status;
}

void TextStreamProbe::featureNumberToCharacter(int code, char ** cbufidx, char * bufstart, int buflen) {
   assert(bufstart-*cbufidx<buflen); // Test array bounds
   int nf = getTargetLayer()->getLayerLoc()->nf;
   if (useCapitalization) {
      assert(nf==97);
      switch(code) {
      case 95:
         **cbufidx = '\n';
         (*cbufidx)++;
         break;
      case 96:
         assert(bufstart-*cbufidx<buflen-1);
         **cbufidx = (char) -62;  // UTF-8 for section sign
         (*cbufidx)++;
         **cbufidx = (char) -89;
         (*cbufidx)++;
         break;
      default:
         **cbufidx = code + 32;
         (*cbufidx)++;
         break;

      }
   }
   else {
      assert(nf==71);
      switch(code) {
      case 95:
         **cbufidx = '\n';
         (*cbufidx)++;
         break;
      case 96:
         assert(bufstart-*cbufidx<buflen-1);
         **cbufidx = (char) -62;  // UTF-8 for section sign
         (*cbufidx)++;
         **cbufidx = (char) -89;
         (*cbufidx)++;
         break;
      default:
         char outcode = code + (char) 32;
         if (outcode>=65 && outcode<=90) outcode += 32;
         **cbufidx = outcode;
         (*cbufidx)++;
         break;
      }
   }

   return;
}


} /* namespace PV */
