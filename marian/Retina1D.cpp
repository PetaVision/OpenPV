/*
 * Retina1DPattern.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: gkenyon
 */

#include "Retina1D.hpp"
#include <stdlib.h>
#include <assert.h>
#include <iostream>

using namespace std;


namespace PV {

//Retina1D::Retina1D() : Retina() {
//}


Retina1D::Retina1D(const char * name, HyPerCol * hc) :
   Retina(name, hc)
{
   mom = (pvdata_t *) malloc(clayer->numNeurons * sizeof(pvdata_t));;
//   createImage(clayer->V);
   createRandomImage(clayer->V);
   cout << "mom: ";
   for (int i = 0; i < clayer->numNeurons; i++) {
     mom[i] = clayer->V[i];
     if (!(i%2)){
        cout << mom[i];
     }
   }
   cout << endl;
   //getchar();

}

Retina1D::~Retina1D()
{
   free(mom);
}

int Retina1D::createImage(pvdata_t * buf) {
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;

   static int t = -1;

   int min = 4; // 16;
   int max = nx - min;

   // slide image left and right by one pixel
   t += 1;
   min += t % 2;
   max += t % 2;

   assert(this->clayer->numFeatures == 2);

   for (int k = 0; k < clayer->numNeurons; k += 2) {
      int kx = kxPos(k, nx, ny, nf);

      int pat = 1;
      if (kx < min || kx > max) {
         pat = 0;
      }
      buf[k]   = 1 - pat;
      buf[k+1] = pat;
   }

   return 0;
}

int Retina1D::createRandomImage(pvdata_t * buf) {
   // break into units of 3 pixels with 8 possible ON/OFF values

   int patterns[8][3];

   patterns[0][0] = 0;
   patterns[0][1] = 0;
   patterns[0][2] = 0;

   patterns[1][0] = 0;
   patterns[1][1] = 0;
   patterns[1][2] = 1;

   patterns[2][0] = 0;
   patterns[2][1] = 1;
   patterns[2][2] = 0;

   patterns[3][0] = 0;
   patterns[3][1] = 1;
   patterns[3][2] = 1;

   patterns[4][0] = 1;
   patterns[4][1] = 0;
   patterns[4][2] = 0;

   patterns[5][0] = 1;
   patterns[5][1] = 0;
   patterns[5][2] = 1;

   patterns[6][0] = 1;
   patterns[6][1] = 1;
   patterns[6][2] = 0;

   patterns[7][0] = 1;
   patterns[7][1] = 1;
   patterns[7][2] = 1;

   assert(this->clayer->numFeatures == 2);

   // TODO - 3 doesn't divide evenly into 64
//   for (int k = 0; k < clayer->numNeurons; k += 6) {
   for (int k = 0; k < clayer->numNeurons; k += 2) {
      //      int pat = rand() % 8;
//      assert(pat < 8);
//      buf[k+0] = patterns[pat][0];
//      buf[k+1] = patterns[pat][1];
//      buf[k+2] = patterns[pat][2];

      int pat = rand() % 2;
      buf[k]   = 1 - pat;
      buf[k+1] = pat;
   }

   return 0;
}


int Retina1D::updateState(float time, float dt)
{
   static int fire = 0;
   static int plot_message = 1;
   static int present_mom = 1;
   static int F = 30; // fire every F steps
   static int P = 1; // present mom every P patterns; the pattern presentation is every F steps
                      // then for F-1 step fire empty

   pvdata_t * V = clayer->V;
   pvdata_t * activity = clayer->activity->data;

   if(time < 8000){ // periodic input - one pattern

     // fire mom every F steps

     if (fire == 0) {
       fire = F;

       for (int k = 0; k < clayer->numNeurons; k++) {
    	   activity[k] = mom[k];
       }


     }
     else {

       fire -= 1;
       for (int k = 0; k < clayer->numNeurons; k++) {
         activity[k] = 0;
       }

     }


   } else { // mixed input: mom every P patterns and random pattern otherwise.

     // present mom every P steps and random patterns
     // otherwise

     if(plot_message){
       plot_message = 0;
       //printf("start alternating mom and random patterns\n");
       //getchar();
     }

     if (fire == 0) {



       if(present_mom==P){

    	   //printf("%d: fire mom\n",fire);
    	   for (int k = 0; k < clayer->numNeurons; k++) {
    		   activity[k] = mom[k];
    	   }
    	   present_mom=0;

       } else {

    	   //printf("%d: fire random\n",fire);
    	   this->createRandomImage(V);
    	   for (int k = 0; k < clayer->numNeurons; k++) {
    		   activity[k] = V[k];
    	   }
    	   present_mom++;

       }

       fire = F;

     }else {

       //printf("%d: fire empty\n",fire);
       fire -= 1;
       for (int k = 0; k < clayer->numNeurons; k++) {
    	   activity[k] = 0;
       }

     }


   } // if train or random phase

   return 0;
}



}
