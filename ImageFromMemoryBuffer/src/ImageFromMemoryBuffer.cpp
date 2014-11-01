/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  Description of the class is in ImageFromMemoryBuffer.hpp
 */

#include "ImageFromMemoryBuffer.hpp"

namespace PV {

ImageFromMemoryBuffer::ImageFromMemoryBuffer(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ImageFromMemoryBuffer::ImageFromMemoryBuffer() {
   initialize_base();
   // protected default constructor; initialize(name,hc) should be called by any derived class's initialization routine
}

int ImageFromMemoryBuffer::initialize_base() {
   buffer = NULL;
   return PV_SUCCESS;
}

int ImageFromMemoryBuffer::initialize(char const * name, HyPerCol * hc) {
   return Image::initialize(name, hc);
}

template <typename pixeltype>
int ImageFromMemoryBuffer::setMemoryBuffer(pixeltype const * externalBuffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, pixeltype zeroval, pixeltype oneval) {
   if (height<=0 || width<=0 || numbands<=0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "ImageFromMemoryBuffer::setMemoryBuffer error: height, width, numbands arguments must be positive.\n");
      }
      return PV_FAILURE;
   }
   imageLoc.nx = height;
   imageLoc.ny = width;
   imageLoc.nf = numbands;
   imageLoc.nxGlobal = height;
   imageLoc.nyGlobal = width;
   imageLoc.kx0 = 0;
   imageLoc.ky0 = 0;
   memset(&imageLoc.halo, 0, sizeof(PVHalo));

   if (parent->columnId()==0) {
      int buffersize = height*width*numbands;
      buffer = (pvadata_t *) malloc((size_t) buffersize * sizeof(pvadata_t));
      if (buffer==NULL) {
         fprintf(stderr, "%s \"%s\": unable to allocate buffer for %d values of %zu chars each: %s\n",
               parent->parameters()->groupKeywordFromName(name), name, buffersize, sizeof(pvadata_t), strerror(errno));
         exit(EXIT_FAILURE);
      }
      for (int k=0; k<buffersize; k++) {
         int x=kxPos(k,width,height,numbands);
         int y=kyPos(k,width,height,numbands);
         int f=featureIndex(k,width,height,numbands);
         int externalIndex = f*bandstride + x*xstride + y*ystride;
         pixeltype q = externalBuffer[externalIndex];
         buffer[k] = pixelTypeConvert(q, zeroval, oneval);
      }
   }
}
template int ImageFromMemoryBuffer::setMemoryBuffer<uint8_t>(uint8_t const * buffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, uint8_t zeroval, uint8_t oneval);

template <typename pixeltype>
pvadata_t ImageFromMemoryBuffer::pixelTypeConvert(pixeltype q, pixeltype zeroval, pixeltype oneval) {
   return ((pvadata_t) (q-zeroval))/((pvadata_t) (oneval-zeroval));
}

int ImageFromMemoryBuffer::initializeActivity() {
   int status = PV_SUCCESS;
   Communicator * icComm = parent->icCommunicator();
   if (parent->columnId()==0) {
      if (buffer == NULL) {
         fprintf(stderr, "%s \"%s\" error: moveBufferToData called without having called setMemoryBuffer.\n",
               parent->parameters()->groupKeywordFromName(name), name);
         exit(PV_FAILURE); // return PV_FAILURE;
      }
      for (int rank=1; rank<icComm->commSize(); rank++) {
         status = moveBufferToData(rank);
         assert(status == PV_SUCCESS);
         MPI_Send(data, getNumExtended(), MPI_FLOAT, rank, 31, icComm->communicator());
      }
      status = moveBufferToData(0);
      assert(status == PV_SUCCESS);
   }
   else {
      MPI_Recv(data, getNumExtended(), MPI_FLOAT, 0, 31, icComm->communicator(), MPI_STATUS_IGNORE);
   }
   return status;
}

int ImageFromMemoryBuffer::moveBufferToData(int rank) {
   if (autoResizeFlag) {
      fprintf(stderr, "autoResizeFlag not implemented for ImageFromMemoryBuffer yet :-(\n");
      exit(EXIT_FAILURE);
   }
   assert(parent->columnId()==0);
   assert(buffer != NULL);
   
   // The part of the buffer with x in [processcolumnnumber * nx, processcolumnnumber * (nx + 1) - 1]
   // and y in [processcolumnnumber * nx, processcolumnnumber * (nx + 1) - 1]
   // gets copied to the restricted space of the given rank.
   // If useImageBCflag is set, we also have to copy as much of the surrounding region as
   // will fit in both the halo and the image.
   // Are we also supposed to mirror any gap between the edge of the image and the edge of the halo?
   // The code doesn't do that at this point.
   // The code below assumes that all processes have the same getLayerLoc().
   Communicator * icComm = parent->icCommunicator();
   int column = columnFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
   int startxbuffer = column * getLayerLoc()->nx;
   int startxdata = getLayerLoc()->halo.lt;
   int row = rowFromRank(rank, icComm->numCommRows(), icComm->numCommColumns());
   int startybuffer = row * getLayerLoc()->ny;
   int startydata = getLayerLoc()->halo.up;
   int xsize = getLayerLoc()->nx;
   int ysize = getLayerLoc()->ny;
   int fsize = getLayerLoc()->nf;
   PVHalo const * halo = &getLayerLoc()->halo;
   if (useImageBCflag) {

      int moveleft = startxbuffer;
      if (halo->lt < moveleft) {
         moveleft = halo->lt;
      }
      if (moveleft > 0) {
         startxbuffer -= moveleft;
         startxdata -= moveleft;
         xsize += moveleft;
      }

      int moveright = imageLoc.nx - (startxbuffer + xsize);
      if (halo->rt < moveright) {
         moveright = halo->rt;
      }
      if (moveright > 0) {
         xsize += moveright;
      }

      int moveup = startybuffer;
      if (halo->up < moveup) {
         moveup = halo->up;
      }
      if (moveup > 0) {
         startybuffer -= moveup;
         startydata -= moveup;
         ysize += moveup;
      }

      int movedown = imageLoc.ny - (startybuffer + ysize);
      if (halo->dn < movedown) {
         movedown = halo->dn;
      }
      if (movedown > 0) {
         ysize += movedown;
      }   
   }
   assert(startxbuffer >= 0 && startxbuffer + xsize <= imageLoc.nxGlobal);
   assert(startybuffer >= 0 && startybuffer + ysize <= imageLoc.nyGlobal);
   if (fsize != 1 && fsize != imageLoc.nf) {
      fprintf(stderr, "%s \"%s\": If nf is greater than 1, or numbands must be the same as nf\n", parent->parameters()->groupKeywordFromName(name), name);
      exit(EXIT_FAILURE);
   }
   if (fsize != imageLoc.nf) {
      assert(fsize == 1);
      // layer has one feature; convert memory buffer to grayscale
      for (int y=0; y<ysize; y++) {
         for (int x=0; x<xsize; x++) {
            int indexdata = kIndex(startxdata+x,startydata+y,0,getLayerLoc()->nx+halo->lt+halo->rt,getLayerLoc()->ny+halo->dn+halo->up,1);
            int indexbuffer = kIndex(startxdata+x,startydata+y,0,imageLoc.nx,imageLoc.ny,imageLoc.nf);
            pvdata_t val = (pvadata_t) 0;
            for (int f=0; f<imageLoc.nf; f++) {
               val += buffer[indexbuffer+f];
            }
            data[indexdata] = val/(pvadata_t) imageLoc.nf;
         }
      }
   }
   else {
      // layer and memory buffer have the same number of features
      for (int y=0; y<ysize; y++) {
         int linestartdata = kIndex(startxdata,startydata+y,0,getLayerLoc()->nx+halo->lt+halo->rt,getLayerLoc()->ny+halo->dn+halo->up,fsize);
         int linestartbuffer = kIndex(startxbuffer,startybuffer+y,0,imageLoc.nx,imageLoc.ny,fsize);
         memcpy(&data[linestartdata], &buffer[linestartbuffer], sizeof(pvadata_t) * xsize * fsize);
      }
   }
   return PV_SUCCESS;
}

ImageFromMemoryBuffer::~ImageFromMemoryBuffer() {
   free(buffer);
}

}  // namespace PV
