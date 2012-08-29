#include "BIDSMovieCloneMap.hpp"


namespace PV{
BIDSMovieCloneMap::BIDSMovieCloneMap(){
   initialize_base();
}

BIDSMovieCloneMap::BIDSMovieCloneMap(const char * name, HyPerCol * hc, int numChannels){
   initialize_base();
   initialize(name, hc, numChannels);
}

int BIDSMovieCloneMap::initialize_base(){
   originalMovie = NULL;
   coords = NULL;
   nxPost = 0;
   nyPost = 0;
   numNodes = 0;
   return PV_SUCCESS;
}

int BIDSMovieCloneMap::initialize(const char * name, HyPerCol * hc, int numChannels){
   HyPerLayer::initialize(name, hc, numChannels);

   //Grab Orig Movie
   const char * strOriginalMovie = (int)(parent->parameters()->stringValue(name, "originalMovie"));
   originalMovie = getParent()->getLayerFromName(strOriginalMovie);
   nbPre = originalMovie->getLayerLoc()->nb;
   nxPre = originalMovie->getLayerLoc()->nx;
   nyPre = originalMovie->getLayerLoc()->ny;
   nf = originalMovie->getLayerLoc()->nf;

   //Grab params
   float nxScale = (float)(parent->parameters()->value(name, "nxScale"));
   float nyScale = (float)(parent->parameters()->value(name, "nyScale"));
   int HyPerColx = (int)(parent->parameters()->value("column", "nx"));
   int HyPerColy = (int)(parent->parameters()->value("column", "ny"));
   numNodes = (nxScale * HyPerColx) * (nyScale * HyPerColy);
   nxPost = nxScale * HyPerColx;
   nyPost = nyScale * HyPerColy;
   coords = BIDSCoords [nxPost] [nyPost];
   int jitter = (int)(parent->parameters()->value(name, "jitter"));

   //Check jitter
   assert(2 * jitter < nbPre);
   assert(jitter >= 0); //jitter cannot be below zero

   //Apply jitter
   setCoords(numNodes, coords, jitter, nxScale, nyScale, HyPerColx, HyPerColy);

   return PV_SUCCESS;
}

void BIDSMovieCloneMap::setCoords(int numNodes, BIDSCoords ** coords, int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy){
   srand(time(NULL));

   int patchSizex = (1/nxScale); //the length of a side of a patch in the HyPerColumn
   int patchSizey = (1/nyScale); //the length of a side of a patch in the HyPerColumn
   int jitterRange = jitter * 2;

   //TODO: Set up physical position for margin nodes
   int i = 0;
   int j = 0;
   for(int lowerboundx = 0; lowerboundx < HyPerColx; lowerboundx = lowerboundx + patchSizex){
      for(int lowerboundy = 0; lowerboundy < HyPerColy; lowerboundy = lowerboundy + patchSizey){
         int jitX = 0;
         int jitY = 0;
         if(jitter > 0){ //else, the nodes should be placed in the middle of each patch
            jitX = rand() % jitterRange - jitter; //stores the x coordinate into the current BIDSCoord structure
            jitY = rand() % jitterRange - jitter; //stores the y coordinate into the current BIDSCoord structure
         }
         coords[i][j].xCoord = lowerboundx + (patchSizex / 2) + jitX; //stores the x coordinate into the current BIDSCoord structure
         coords[i][j].yCoord = lowerboundy + (patchSizey / 2) + jitY; //stores the y coordinate into the current BIDSCoord structure
         j++;
      }
      i++;
      j = 0;
   }

   //for(int i = 0; i < numNodes; i++){
      //printf("[x,y] = [%d,%d]\n", coords[i].xCoord, coords[i].yCoord);
   //}
}

BIDSCoords BIDSMovieCloneMap::getCoords(int x, int y){
   return coords[x][y];
}

int BIDSMovieCloneMap::updateState(float timef, float dt){
   //Get output buffer
   pvdata_t * output = getCLayer()->V;
   pvdata_t * input = originalMovie->getCLayer()->activity->data;
   int indexPre;
   int indexPost;
   BIDSCoords coord;
   //Iterate through post layer
   for (int i = 0; i < nxPost; i++){
      for (int j = 0; j < nyPost; j++){
         //Iterate through features
         for (int k = 0; k < nf; k++){
            coord = coords[i][j];
            indexPre = kIndex(coord.xCoord+nbPre, coord.yCoord+nbPre, k, nxPre+2*nbPre, nyPre+2*nbPre, nf);
            indexPost = kIndex(i, j, k, nxPost, nyPost, nf);
            output[indexPost] = input[indexPre];
         }
      }
   }
}


}
