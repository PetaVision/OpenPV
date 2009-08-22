/*
 * CocircConn.cpp
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#include "CocircConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

CocircConn::CocircConn(const char * name,
                       HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
{
   this->connId = hc->numberOfConnections();
   this->name = strdup(name);
   this->parent = hc;
   this->numAxonalArborLists = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

int CocircConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();
      const float sigma    = params->value(name, "sigma");
      const float rMax     = params->value(name, "rMax");
      const float strength = params->value(name, "strength");

      float r2Max = rMax * rMax;

      const int xScale = pre->clayer->xScale;
      const int yScale = pre->clayer->yScale;

      int nfPre = pre->clayer->numFeatures;

      const int borderId = 0;
      const int numPatches = numberOfWeightPatches(borderId);
      for (int i = 0; i < numPatches; i++) {
         int fPre = i % nfPre;
         cocircWeights(wPatches[borderId][i], fPre, xScale, yScale, sigma, r2Max, strength);
      }
   }
   else {
      int err = 0;
      size_t size, count, dim[3];

       // TODO rank 0 should read and distribute file

      FILE * fd = fopen(filename, "rb");

      if (fd == NULL) {
         fprintf(stderr,
                 "FileConn:: couldn't open file %s, using 8x8 weights = 1\n",
                  filename);
         return -1;
      }

      if ( fread(dim,    sizeof(size_t), 3, fd) != 3 ) err = 1;
      if ( fread(&size,  sizeof(size_t), 1, fd) != 1 ) err = -1;
      if ( fread(&count, sizeof(size_t), 1, fd) != 1 ) err = -1;

      // check for consistency

      if (dim[0] != (size_t) nfp) err = -1;
      if (dim[1] != (size_t) nxp) err = -1;
      if (dim[2] != (size_t) nyp) err = -1;
      if ((int) count != numAxonalArborLists) err = -1;
      if (size  != sizeof(PVPatch) + nxp*nyp*nfp*sizeof(float) ) err = -1;

      if (err) {
         fprintf(stderr, "FileConn:: ERROR: difference in dim, size or count of patches\n");
         return err;
      }

      for (unsigned int i = 0; i < count; i++) {
         PVPatch* patch = wPatches[0][i];
         if ( fread(patch, size, 1, fd) != 1) {
            fprintf(stderr, "FileConn:: ERROR reading patch %d\n", i);
            return -1;
         }
         // TODO fix address with a function
         patch->data = (pvdata_t*) ((char*) patch + sizeof(PVPatch));
      }

      // TODO - adjust strides sy, sf in weight patches
   } // end if for filename

   return 0;
}

int CocircConn::cocircWeights(PVPatch * wp, int fPre, int xScale, int yScale,
                              float sigma, float r2Max, float strength)
{
   float gDist = 0.0;

   pvdata_t * w = wp->data;

   // get parameters

   PVParams * params = parent->parameters();

   const int nfPre = (int) params->value(pre->getName(), "nf");
   const int nfPst = (int) params->value(post->getName(), "nf");
   const int noPre = (int) params->value(pre->getName(), "no");
   const int noPst = (int) params->value(post->getName(), "no");

   // from pv_common.h
   // DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
   const float dKv          = params->value(name, "deltaCurvature");

   const float sigma_cocirc = params->value(name, "sigma_cocirc");
   const float sigma_kurve  = params->value(name, "sigma_kurve");

   float rotate = 1;  // rotate/offset so orientations aren't aligned with axis
   if (params->present(name, "rotate")) rotate = params->value(name, "rotate");
   float deltaThetaMax = PI/2.0;
   if (params->present(name, "deltaThetaMax")) deltaThetaMax = params->value(name, "deltaThetaMax");
   float cocirc_self = pre != post;
   if (params->present(name, "cocirc_self")) cocirc_self = params->value(name, "cocirc_self");

   const float sigma2        = 2 * sigma * sigma;
   const float sigma_cocirc2 = 2 * sigma_cocirc * sigma_cocirc;
   const float sigma_kurve2  = 2 * sigma_kurve  * sigma_kurve;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   // strides
   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   const float dx = powf(2, xScale);
   const float dy = powf(2, yScale);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   const float x0 = -(nx/2.0 - 0.5) * dx;
   const float y0 = +(ny/2.0 - 0.5) * dy;

   const int nKurvePre = nfPre / noPre;
   const int nKurvePst = nfPst / noPst;

   const float dTh = PI/noPst;
   const float th0 = rotate*dTh/2.0;

   const int iKvPre = fPre % nKurvePre;
   const int iThPre = fPre / nKurvePre;

   const float kurvePre = 0.0 + iKvPre * dKv;
   const float thetaPre = th0 + iThPre * dTh;

   // loop over all post synaptic neurons in patch
   for (int f = 0; f < nf; f++) {
      int iKvPst = f % nKurvePst;
      int iThPst = f / nKurvePst;

      float kurvePst = 0.0 + iKvPst * dKv;
      float thetaPst = th0 + iThPst * dTh;

      float deltaTheta = RAD_TO_DEG * fabs(thetaPre - thetaPst);
      deltaTheta = deltaTheta <= 90. ? deltaTheta : 180. - deltaTheta;
      if (deltaTheta > deltaThetaMax) {
         continue;
      }

      float gCocirc   = 1.0;
      float gKurvePre = 1.0;
      float gKurvePst = 1.0;

      for (int j = 0; j < ny; j++) {
         float y = y0 - j * dy;
         for (int i = 0; i < nx; i++) {
            float x  = x0 + i*dx;
            float d2  = x * x + y * y;
            if (d2 > r2Max) continue;  // || (d2 < r2Min)

            gDist = expf(-d2/sigma2);

            if (d2 == 0 && (cocirc_self == 0)  ) {
               // TODO - why calculate anything else
               gDist = 0.0;
               gCocirc = sigma_cocirc > 0 ?
                         expf(-deltaTheta * deltaTheta / sigma_cocirc2 ) :
                         expf(-deltaTheta * deltaTheta / sigma_cocirc2 ) - 1.0;
                if ( (nKurvePre > 1) && (nKurvePst > 1) ) {
                    gKurvePre = expf( -(kurvePre - kurvePst) * (kurvePre - kurvePst) / sigma_kurve2 );
                }
            }
            else {
                float dxP = + x * cos(thetaPre) + y * sin(thetaPre);
                float dyP = - x * sin(thetaPre) + y * cos(thetaPre);

                // The first version implements traditional association field
                // of Field, Li, Zucker, etc. It looks like a bowtie for most
                // orientations, although orthogonal angles are supported at
                // 45 degrees in all four quadrants.

                float atanx2 = thetaPre + 2. * atan2f(dyP, dxP);    // preferred angle (rad)
                atanx2 += 2. * PI;
                atanx2 = fmod( atanx2, PI );
                float chi = RAD_TO_DEG * fabs( atanx2 - thetaPst ); // degrees
                if ( chi >= 90. ) {
                    chi = 180. - chi;
                }
                if ( noPre > 1 && noPst > 1 ) {
                    gCocirc = sigma_cocirc2 > 0 ?
                        expf(-chi * chi / sigma_cocirc2 ) :
                        expf(-chi * chi / sigma_cocirc2 ) - 1.0;
                }

                float cocircKurve = fabs(2 * dyP) / d2;
                gKurvePre = ( nKurvePre > 1 ) ?
                    exp( -pow( (cocircKurve - fabs(kurvePre)), 2 ) / sigma_kurve2 ) :
                    1.0;
                gKurvePst =  ( (nKurvePre > 1) && (nKurvePst > 1) && (sigma_cocirc2 > 0) ) ?
                    exp( -pow( (cocircKurve - fabs(kurvePst)), 2 ) / sigma_kurve2 ) :
                    1.0;
            }

            w[i*sx + j*sy + f*sf] = gDist * gKurvePre * gKurvePst * gCocirc;

         }
      }
   }

   // normalize
   for (int f = 0; f < nf; f++) {
      float sum = 0;
      for (int i = 0; i < nx*ny; i++) sum += w[f + i*nf];

      if (sum == 0.0) continue;  // all weights == zero is ok

      float factor = strength/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

}
