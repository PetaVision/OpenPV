/*
 * pv.cpp
 *
 */

#include <src/columns/buildandrun.hpp>
#include "ChannelProbe.hpp"
#include "RandomPatchMovie.hpp"
#include "ShadowRandomPatchMovie.hpp"
#include "RandomPatchMovieProbe.hpp"
#include <unistd.h>
#ifdef PV_USE_MPI
   #include <mpi.h>
#else
   #include <src/include/mpi_stubs.h>
#endif // PV_USE_MPI

void * customgroups(const char * keyword, const char * name, HyPerCol * hc);
int printarch();

int main(int argc, char * argv[]) {
#ifdef PV_USE_MPI
   int mpi_is_initialized;
   MPI_Initialized(&mpi_is_initialized);
   if( !mpi_is_initialized ) MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
   int rank = 0;
#endif // PV_USE_MPI
   int charhit;
   int status = PV_SUCCESS;
   fflush(stdout);
   if( rank == 0 ) {
      printarch();
      if( argc > 1 ) {
         printf("Hit enter to begin! ");
         fflush(stdout);
         charhit = getc(stdin);
      }
   }
#ifdef PV_USE_MPI
      int ierr;
      ierr = MPI_Bcast(&charhit, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif // PV_USE_MPI
      status = buildandrun(argc, argv, NULL, NULL, &customgroups);
#ifdef PV_USE_MPI
   if( !mpi_is_initialized) MPI_Finalize();
#endif // PV_USE_MPI
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   int status;
   PVParams * params = hc->parameters();
   HyPerLayer * targetlayer;
   const char * message;
   const char * filename;
   if( !strcmp( keyword, "ChannelProbe") ) {
      status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer, &message, &filename);
      if(status != PV_SUCCESS) {
         fprintf(stderr, "Skipping params group \"%s\"\n", name);
         return NULL;
      }
      if(filename == NULL) {
         fprintf(stderr, "ChannelProbe \"%s\": parameter filename must be set.  Skipping group.\n", name);
         return NULL;
      }
      ChannelType channelCode;
      int channelNo = params->value(name, "channelCode", -1);
      if( decodeChannel( channelNo, &channelCode ) != PV_SUCCESS) {
         fprintf(stderr, "%s \"%s\": parameter channelCode must be set.\n", keyword, name);
         return NULL;
      }
      else {
         ChannelProbe * channelProbe = new ChannelProbe(filename, hc, channelCode);
         assert(targetlayer);
         if( channelProbe ) targetlayer->insertProbe(channelProbe);
         checknewobject((void *) channelProbe, keyword, name);
         return (void *) channelProbe;
      }
   }
   if( !strcmp( keyword, "RandomPatchMovie") ) {
      RandomPatchMovie * addedLayer;
      const char * imagelabelspath = hc->parameters()->stringValue(name, "imageListPath");
      if (imagelabelspath) {
         addedLayer = new RandomPatchMovie(name, hc, imagelabelspath);
      }
      else {
         fprintf(stderr, "Group \"%s\": Parameter group for class RandomPatchMovie must set string parameter imageListPath\n", name);
         addedLayer = NULL;
      }
      return (void *) addedLayer;
   }
   if( !strcmp( keyword, "ShadowRandomPatchMovie") ) {
      RandomPatchMovie * addedLayer;
      const char * imagelabelspath = hc->parameters()->stringValue(name, "imageListPath");
      if (imagelabelspath) {
         addedLayer = new ShadowRandomPatchMovie(name, hc, imagelabelspath);
      }
      else {
         fprintf(stderr, "Group \"%s\": Parameter group for class ShadowRandomPatchMovie must set string parameter imageListPath\n", name);
         addedLayer = NULL;
      }
      return (void *) addedLayer;
   }
   if( !strcmp( keyword, "RandomPatchMovieProbe") ) {
      targetlayer = getLayerFromParameterGroup(name, hc, "targetLayer");
      if( ! targetlayer ) {
         fprintf(stderr, "Group \"%s\": Class %s must define targetLayer\n", name, keyword);
         return NULL;
      }
      const char * filename = getStringValueFromParameterGroup(name, params, "probeOutputFile", false);
      if( ! filename ) {
         fprintf(stderr, "Group \"%s\": Class %s must define probeOutputFile\n", name, keyword);
         return NULL;
      }
      RandomPatchMovieProbe * rpmProbe = new RandomPatchMovieProbe(filename, hc, name);
      if( rpmProbe ) targetlayer->insertProbe(rpmProbe);
      return (void *) rpmProbe;
   }
   fprintf(stderr, "Group \"%s\": Keyword \"%s\" unrecognized.  Skipping group.\n", name, keyword);
   // TODO smarter error handling
   return NULL;
}

int printarch() {
   printf("This is PetaVision version something point something-or-other.\n");
   const char * formatstr = "%s is%s set\n";
   const char * notset = " not";
#ifdef PV_ARCH_64
   printf(formatstr,"PV_ARCH_64","");
#else // PV_ARCH_64
   printf(formatstr,"PV_ARCH_64",notset);
#endif // PV_ARCH_64

#ifdef PV_USE_MPI
   printf(formatstr,"PV_USE_MPI","");
#else // PV_USE_MPI
   printf(formatstr,"PV_USE_MPI",notset);
#endif // PV_USE_MPI

#ifdef PV_USE_OPENCL
   printf(formatstr,"PV_USE_OPENCL","");
#else // PV_USE_OPENCL
   printf(formatstr,"PV_USE_OPENCL",notset);
#endif // PV_USE_OPENCL

#ifdef PV_USE_OPENGL
   printf(formatstr,"PV_USE_OPENGL","");
#else // PV_USE_OPENGL
   printf(formatstr,"PV_USE_OPENGL",notset);
#endif // PV_USE_OPENGL

#ifdef PV_USE_GDAL
   printf(formatstr,"PV_USE_GDAL","");
#else // PV_USE_GDAL
   printf(formatstr,"PV_USE_GDAL",notset);
#endif // PV_USE_GDAL

#ifdef PV_USE_PTHREADS
   printf(formatstr,"PV_USE_PTHREADS","");
#else // PV_USE_PTHREADS
   printf(formatstr,"PV_USE_PTHREADS",notset);
#endif // PV_USE_PTHREADS

#ifdef IBM_CELL_BE
   printf(formatstr,"IBM_CELL_BE","");
#else // IBM_CELL_BE
   printf(formatstr,"IBM_CELL_BE",notset);
#endif // IBM_CELL_BE

   return PV_SUCCESS;
}
