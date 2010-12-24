/*
 * RandomConn.cpp
 *
 *  Created on: Apr 27, 2009
 *      Author: rasmussn
 */

#include "RandomConn.hpp"
#include "../utils/pv_random.h"
#include <assert.h>
#include <string.h>
#include <time.h>

namespace PV {

RandomConn::RandomConn(const char * name,
                       HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel)
{
   HyPerConn::initialize_base();

   randDistType = UNIFORM; //Uniform distribution is the default
   HyPerConn::initialize(name, hc, pre, post, channel);
}

RandomConn::RandomConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
                       HyPerLayer * post, ChannelType channel, RandDistType distrib)
{
   initialize_base();

   randDistType = distrib;
   initialize(name, hc, pre, post, channel);
}

int RandomConn::initializeRandomWeights(unsigned long seed)
{
   switch(randDistType) {
   case UNIFORM:
      return initializeUniformWeights(seed);
   case GAUSSIAN:
      return initializeGaussianWeights(seed);
   default:
      fprintf(stderr, "RandomConn: Warning: Unknown distribution type. "
                      "Using Uniform distribution.\n");
      return initializeUniformWeights(seed);
   }
}

int RandomConn::initializeUniformWeights(unsigned long seed)
{
   PVParams * params = parent->parameters();

   wMin = (long) params->value(name, "wMin", 0.0f);
   idum = (long) params->value(name, "idum", -1.0f);

   float wMinInit = params->value(name, "wMinInit", wMin);
   float wMaxInit = params->value(name, "wMaxInit", wMax);

   const int arbor = 0;
   const int numPatches = numWeightPatches(arbor);
   for (int k = 0; k < numPatches; k++) {
      uniformWeights(wPatches[arbor][k], wMinInit, wMaxInit, seed);
   }

   return 0;
}

/**
 * calculate random weights for a patch given a range between wMin and wMax
 */
int RandomConn::uniformWeights(PVPatch * wp, float wMin, float wMax, unsigned long seed)
{
   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;
   const int nk = nx * ny * nf;

   double p = (wMax - wMin) / pv_random_max();

   // loop over all post-synaptic cells in patch
   for (int k = 0; k < nk; k++) {
      w[k] = (float) (wMin + p * pv_random());
   }

   return 0;
}

int RandomConn::initializeGaussianWeights(unsigned long seed)
{
   PVParams * params = parent->parameters();


   if (params->present(name, "wGaussMean")){
      wGaussMean = params->value(name, "wGaussMean");
   }
   else {
      wGaussMean = (wMin + wMax) / 2.0f;
   }

   if (params->present(name, "wGaussStdev")){
      wGaussStdev = params->value(name, "wGaussStdev");
   }
   else {
      wGaussStdev = 1.0;
   }

   idum = (long) params->value(name, "idum", -1.0f);

   const int arbor = 0;
   const int numPatches = numWeightPatches(0);
   printf("numPatches = %d  wGaussMean = %f wGaussStdev = %f idum = %ld\n",
         numPatches, wGaussMean , wGaussStdev, idum);

   for (int k = 0; k < numPatches; k++) {
      //gaussianWeights(wPatches[k], wGaussMean, wGaussStdev, seed);
      gaussianWeightsMA(wPatches[arbor][k], wGaussMean, wGaussStdev, &idum); // ma
   }

   return 0;
}

/**
 * calculate random weights with Gaussian distribution for a patch given
 * a mean (mean) and standard deviation (stdev)
 *
 * Returns 0 if successful, else non-zero.
 */
int RandomConn::gaussianWeights(PVPatch *wp, float mean, float stdev, unsigned long seed)
{
   if ((NULL == wp) || (NULL == wp->data)) {
      fprintf(stderr, "HyPerConn: Error reading patch.\n");
      return -1;
   }

   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;
   const int nk = nx * ny * nf;

   srand(seed);

   // loop over all post-synaptic cells in patch
   for (int k = 0; k < nk; k++) {
      w[k] = randgauss(mean, stdev);
   }

   return 0;
}
/*
 * This uses a seed (idum) that is read from the params file and it is passed as an argument
 * to the random number generator. We use ran1 from Numerical Recipies.
 *
 */
int RandomConn::gaussianWeightsMA(PVPatch *wp, float mean, float stdev, long *idum)
{
   if ((NULL == wp) || (NULL == wp->data)) {
      fprintf(stderr,"HyPerConn: Error reading patch.\n");
      return -1;
   }

   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;
   const int nk = nx * ny * nf;

   // loop over all post-synaptic cells in patch
   for (int k = 0; k < nk; k++) {
      w[k] = randgaussMA(mean, stdev, idum);
      if (w[k] < 0.0)
         w[k] = 0.0;
   }

   return 0;
}


/**
 * generate random numbers with Gaussian distribution given
 * mean (mean) and standard deviation (stdev)
 * ToDo: Code source: http://www.taygeta.com/random/gaussian.html
 */
float RandomConn::randgauss(float mean, float stdev)
{
   float urandx1, urandx2, w, y1;
   static float y2;
   static int use_last = 0;

   if (use_last)                   /* use value from previous call */
   {
      y1 = y2;
      use_last = 0;
   }
   else
   {
     do {
         urandx1 = 2.0f * pv_random_prob() - 1.0f;
         urandx2 = 2.0f * pv_random_prob() - 1.0f;
         w = urandx1 * urandx1 + urandx2 * urandx2;
      } while (w >= 1.0f);

      w = sqrtf((-2.0f * log(w)) / w);
      y1 = urandx1 * w;
      y2 = urandx2 * w;
      use_last = 1;
   }
   return(mean + y1 * stdev);
}

float RandomConn::randgaussMA(float mean, float stdev, long *idum)
{
   float fac, rsq, v1, v2;
   static float gset;
   static int iset = 0;

   if (iset == 0)                   /* use value from previous call */
   {
      do {
         v1 = 2.0f * ran1(idum) - 1.0f;
         v2 = 2.0f * ran1(idum) - 1.0f;
         rsq = v1 * v1 + v2 * v2;
      } while (rsq >= 1.0f || rsq == 0.0f);

      fac = sqrt((-2.0f * log(rsq)) / rsq);
      gset = mean + v1 * fac * stdev;
      iset = 1;
      return mean + v2 * fac * stdev;
   } else {
      iset = 0;
      return gset;
   }

}


#define IA 16807
#define IM 2147483647
#define AM (1.0f/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0f-EPS)

float RandomConn::ran1(long *idum)
{
        int j;
        long k;
        static long iy=0;
        static long iv[NTAB];
        float temp;

        if (*idum <= 0 || !iy) {
                if (-(*idum) < 1) *idum=1;
                else *idum = -(*idum);
                for (j=NTAB+7;j>=0;j--) {
                        k=(*idum)/IQ;
                        *idum=IA*(*idum-k*IQ)-IR*k;
                        if (*idum < 0) *idum += IM;
                        if (j < NTAB) iv[j] = *idum;
                }
                iy=iv[0];
        }
        k=(*idum)/IQ;
        *idum=IA*(*idum-k*IQ)-IR*k;
        if (*idum < 0) *idum += IM;
        j=iy/NDIV;
        iy=iv[j];
        iv[j] = *idum;
        if ((temp=AM*iy) > RNMX) return RNMX;
        else return temp;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX


}
