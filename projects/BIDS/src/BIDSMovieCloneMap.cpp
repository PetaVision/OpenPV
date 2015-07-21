#include "BIDSMovieCloneMap.hpp"


namespace PV{
BIDSMovieCloneMap::BIDSMovieCloneMap(){
   initialize_base();
}

BIDSMovieCloneMap::BIDSMovieCloneMap(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

int BIDSMovieCloneMap::initialize_base(){
   numChannels = 0;
   originalMovie = NULL;
   coords = NULL;
   nxPost = 0;
   nyPost = 0;
   return PV_SUCCESS;
}

int BIDSMovieCloneMap::initialize(const char * name, HyPerCol * hc){
   HyPerLayer::initialize(name, hc);

   //Check jitter
//   assert(2 * jitter < nbPre);
   if(jitter < 0) {
      fprintf(stderr, "Error in BIDSMovieCloneMap \"%s\": jitter cannot be below zero.\n", name);
      abort();
   }

   return PV_SUCCESS;
}

int BIDSMovieCloneMap::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_originalMovie(ioFlag);
   ioParam_jitter(ioFlag);
   return PV_SUCCESS;
}

void BIDSMovieCloneMap::ioParam_originalMovie(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalMovie", &originalMovieName);
}

void BIDSMovieCloneMap::ioParam_jitter(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "jitter", &jitter);
}

int BIDSMovieCloneMap::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalMovie = getParent()->getLayerFromName(originalMovieName);

   int HyPerColx = parent->getNxGlobal();
   int HyPerColy = parent->getNyGlobal();
   nxPost = (int)(nxScale * HyPerColx);
   nyPost = (int)(nyScale * HyPerColy);

   return status;
}

int BIDSMovieCloneMap::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   //Grab params
   int HyPerColx = parent->getNxGlobal();
   int HyPerColy = parent->getNyGlobal();

   int numNodes = nxPost * nyPost;
   coords = (BIDSCoords*)malloc(sizeof(BIDSCoords) * numNodes);

   //Apply jitter
   setCoords(jitter, nxScale, nyScale, HyPerColx, HyPerColy);

   return status;
}

void BIDSMovieCloneMap::setCoords(int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy){
   int patchSizex = (int)(1/nxScale); //the length of a side of a patch in the HyPerColumn
   int patchSizey = (int)(1/nyScale); //the length of a side of a patch in the HyPerColumn
   int jitterRange = jitter * 2;

   //TODO: Set up physical position for margin nodes
   int i  = 0;
   for(int lowerboundy = 0; lowerboundy < HyPerColy; lowerboundy = lowerboundy + patchSizey){
      for(int lowerboundx = 0; lowerboundx < HyPerColx; lowerboundx = lowerboundx + patchSizex){
         int jitX = 0;
         int jitY = 0;
         if(jitter > 0){ //else, the nodes should be placed in the middle of each patch
            jitX = rand() % jitterRange - jitter; //stores the x coordinate into the current BIDSCoord structure
            jitY = rand() % jitterRange - jitter; //stores the y coordinate into the current BIDSCoord structure
         }
         coords[i].xCoord = lowerboundx + (patchSizex / 2) + jitX; //stores the x coordinate into the current BIDSCoord structure
         coords[i].yCoord = lowerboundy + (patchSizey / 2) + jitY; //stores the y coordinate into the current BIDSCoord structure
         i++;
      }
   }

}

BIDSCoords* BIDSMovieCloneMap::getCoords(){
   return coords;
}

int BIDSMovieCloneMap::getNxOrig() {return originalMovie->getLayerLoc()->nx;}
int BIDSMovieCloneMap::getNyOrig() {return originalMovie->getLayerLoc()->ny;}
int BIDSMovieCloneMap::getNfOrig() {return originalMovie->getLayerLoc()->nf;}
PVHalo const * BIDSMovieCloneMap::getHaloOrig() {return &(originalMovie->getLayerLoc()->halo);}

int BIDSMovieCloneMap::updateState(double timef, double dt){
   //Get output buffer
   pvdata_t * output = getCLayer()->V;
   pvdata_t * input = originalMovie->getCLayer()->activity->data;
   int indexPre;
   int indexPost;
   BIDSCoords coord;
   //Iterate through post layer
   int nxPre = getNxOrig();
   int nyPre = getNyOrig();
   int nf = getNfOrig();
   PVHalo const * haloPre = getHaloOrig();
   for (int i = 0; i < nxPost * nyPost; i++){
      //Iterate through features
      for (int k = 0; k < nf; k++){
         coord = coords[i];
         indexPre = kIndex(coord.xCoord+haloPre->lt, coord.yCoord+haloPre->up, k, nxPre+haloPre->lt+haloPre->rt, nyPre+haloPre->dn+haloPre->up, nf);
         int xPost = i % nxPost;
         int yPost = (int) i/nxPost;
         indexPost = kIndex(xPost, yPost, k, nxPost, nyPost, nf);
         output[indexPost] = input[indexPre];
//         std::cout << "Frame number " << timef * dt << " Node (" << coord.xCoord << "," << coord.yCoord << "):  " << input[indexPre] << "\n";
      }
   }
   HyPerLayer::setActivity();

   return PV_SUCCESS;
}

int BIDSMovieCloneMap::getNumNodes(){
   return nxPost * nyPost;
}

BIDSMovieCloneMap::~BIDSMovieCloneMap(){
   free(originalMovieName);
   free(coords);
}
}
