/*
 * test_gs_io.cpp
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/layers/HyPerLayer.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/layers/fileread.h"
#include "../src/io/io.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

#undef DEBUG_OUTPUT

int createTestFile(const char* filename, int count, float* buf);
int testOutput(const char* filename, PV::HyPerLayer* l, float* inBuf, float* outBuf);

const char filename[] = "test_layer_direct.bin";
const char outfile[]  = "test_layer_direct_out.bin";

int main(int argc, char* argv[])
{
    int err = 0;
    float *V, *inBuf, *outBuf;
    fileread_params params;

    int count = NX*NY;

    inBuf  = (float*) malloc(count*sizeof(float));
    outBuf = (float*) malloc(count*sizeof(float));

    params.invert = 0.0;
    params.uncolor = 0.0; // if true, pixels>0 become 1.0
    //    params.IMax = 0.0;    // intensity for on pixels
    //    params.poisson = 0.0; // spike as poisson?
    params.poissonEdgeProb = 0.0;  // if so, prob
    params.poissonBlankProb = 0.0; // spike as poisson?
    params.beginStim = 0.0;
    params.endStim = 0.0;
    params.filename = filename;

    PV::HyPerCol* hc = new PV::HyPerCol("column", argc, argv);

    if (hc->columnId() == 0) {
       err = createTestFile(filename, count, inBuf);
       if (err != 0) {
          fprintf(stderr, "[%d]: ERROR - WARNING - exiting without MPI_Finalize()",
                  hc->columnId());
          exit(err);
       }
    }

    // construct layers
    PV::HyPerLayer* Retina  = new PV::Retina("Test Retina", hc, filename);

    Retina->setParams(9, sizeof(params), (float*) &params);

    hc->initFinish();

    V = Retina->clayer->V;

#ifdef DEBUG_OUTPUT
    fprintf(stderr, "[%d]: about to gather\n", hc->columnId());
#endif
    err = gatherWriteFile(outfile, Retina->clayer, V, MPI_COMM_WORLD);
    if (hc->columnId() == 0) {
       err = testOutput(outfile, Retina, inBuf, outBuf);
    }

#ifdef DEBUG_OUTPUT
    for (int i = 0; i < 65; i++) {
      fprintf(stderr, "[%d]: [%d]=%f\n", hc->columnId(), i, V[i]);
    }

    if (hc->columnId() == 0) {
       for (int i = 0; i < 65; i++) {
	  fprintf(stderr, "[%d]: [%d]=%f\n", hc->columnId(), i, outBuf[i]);
       }
    }
    fprintf(stderr, "[%d]: barrier\n", hc->columnId());
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    free(inBuf);
    free(outBuf);

    delete hc;

    return err;
}


int testOutput(const char* filename, PV::HyPerLayer* l, float* inBuf, float* outBuf)
{
   int result, err = 0;
   int rank = l->clayer->columnId;
   int nTotal = (int) (l->clayer->loc.nxGlobal * l->clayer->loc.nyGlobal);

   FILE* fd = fopen(filename, "rb");
   if (fd == NULL) {
      err = -1;
      fprintf(stderr, "[%d]: ERROR: testOutput: couldn't open file %s\n", rank, filename);
      return err;
   }

   result = fread(outBuf, sizeof(float), (size_t) nTotal, fd);
   fclose(fd);
   if (result != nTotal) {
      fprintf(stderr, "[ ]: testOutput: ERROR writing to file %s\n", filename);
   }

   for (int i = 0; i < nTotal; i++) {
      if (inBuf[i] != outBuf[i]) {
	 err = 1;
	 fprintf(stderr, "[%d]: ERROR: testOutput: buffers differ at %d\n", rank, i);
	 return err;
      }
   }

   return err;
}


int createTestFile(const char* filename, int nTotal, float* buf)
{
    int i, err = 0;
    size_t result;

    FILE* fd = fopen(filename, "wb");

    for (i = 0; i < nTotal; i++) {
       buf[i] = (float) i;
    }

    result = fwrite(buf, sizeof(float), nTotal, fd);
    fclose(fd);
    if (result != (size_t) nTotal) {
       fprintf(stderr, "[ ]: createTestFile: ERROR writing to file %s\n", filename);
    }

    fd = fopen(filename, "rb");
    result = fread(buf, sizeof(float), nTotal, fd);
    fclose(fd);
    if (result != (size_t) nTotal) {
       fprintf(stderr, "[ ]: createTestFile: ERROR reading from file %s\n", filename);
    }

    err = 0;
    for (i = 0; i < nTotal; i++) {
        if (buf[i] != (float) i) {
	    err = 1;
	    fprintf(stderr, "%s file is incorrect at %d\n", filename, i);
        }
    }

    return err;
}
