/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "ChannelProbe.hpp"
#include "OnlineLearningKConn.hpp"
#include "RandomPatchMovie.hpp"
#include "RandomPatchMovieProbe.hpp"
#include "ShadowRandomPatchMovie.hpp"
#include <unistd.h>
#ifdef PV_USE_MPI
   #include <mpi.h>
#else
   #include <include/mpi_stubs.h>
#endif // PV_USE_MPI

void * customgroups(const char * keyword, const char * name, HyPerCol * hc);
int printarch();

int main(int argc, char * argv[]) {
#ifdef PV_USE_MPI
   int mpi_is_initialized;
   MPI_Initialized(&mpi_is_initialized);
   if( !mpi_is_initialized ) MPI_Init(&argc, &argv);
#endif // PV_USE_MPI
   int status = PV_SUCCESS;
   status = buildandrun(argc, argv, NULL, NULL, &customgroups);
#ifdef PV_USE_MPI
   if( !mpi_is_initialized) MPI_Finalize();
#endif // PV_USE_MPI
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   int status;
   PVParams * params = hc->parameters();
   void * addedGroup;
   if( !strcmp( keyword, "ChannelProbe") ) {
      ChannelProbe * channelProbe = new ChannelProbe(name, hc);
      addedGroup = (void *) channelProbe;
   }
   if( !strcmp( keyword, "OnlineLearningKConn") ) {
      OnlineLearningKConn * addedConn = new OnlineLearningKConn(name, hc);
      addedGroup = (void *) addedConn;
   }
   if( !strcmp( keyword, "RandomPatchMovie") ) {
      RandomPatchMovie * addedLayer = new RandomPatchMovie(name, hc);
      addedGroup = (void *) addedLayer;
   }
   if( !strcmp( keyword, "RandomPatchMovieProbe") ) {
      RandomPatchMovieProbe * rpmProbe = new RandomPatchMovieProbe(name, hc);
      addedGroup = (void *) rpmProbe;
   }
   if( !strcmp( keyword, "ShadowRandomPatchMovie") ) {
      RandomPatchMovie * addedLayer = new ShadowRandomPatchMovie(name, hc);
      addedGroup = (void *) addedLayer;
   }
   checknewobject(addedGroup, keyword, name, hc);
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

#ifdef PVP_DEBUG
   printf(formatstr, "PVP_DEBUG", "");
#else // PVP_DEBUG
   printf(formatstr,"PVP_DEBUG", notset);
#endif // PVP_DEBUG

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
