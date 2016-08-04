/*
 * main.cpp
 */


#include <columns/buildandrun.hpp>

int checkoutput(HyPerCol * hc, int argc, char ** argv);
//checkoutput is passed as a custom handle in the buildandrun customexit argument,
// so that it is called after HyPerCol::run but before the HyPerCol is deleted.

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &checkoutput);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkoutput(HyPerCol * hc, int argc, char ** argv) {
   // This should really go in a probe so it can check every timestep.
   // Column should have two layers and one connection

   int status = PV_SUCCESS;
   pvErrorIf(!(hc->numberOfLayers()==2 && hc->numberOfConnections()==1), "Test failed.\n");

   // Input layer should be 2x2 with values 1, 2, 3, 4;
   // and have margin width 1 with mirror boundary conditions off.
   HyPerLayer * inLayer = hc->getLayer(0);
   const PVLayerLoc * inLoc = inLayer->getLayerLoc();
   pvErrorIf(!(inLoc->nxGlobal==2 && inLoc->nyGlobal==2 && inLoc->nf==1), "Test failed.\n");
   assert(inLoc->halo.lt==1 &&
          inLoc->halo.rt==1 &&
          inLoc->halo.dn==1 &&
          inLoc->halo.up==1 &&
          inLayer->getNumGlobalExtended()==16);
   
   pvInfo().flush();
   MPI_Barrier(hc->getCommunicator()->communicator());
   for (int r=0; r<hc->getCommunicator()->commSize(); r++) {
      if (r==hc->columnId()) {
         pvInfo().printf("Rank %d, Input layer activity\n",r);
         for (int k=0; k<inLayer->getNumExtended(); k++) {
            int x=kxPos(k,inLoc->nx+inLoc->halo.lt+inLoc->halo.rt,inLoc->ny+inLoc->halo.dn+inLoc->halo.up,inLoc->nf)-inLoc->halo.lt+inLoc->kx0;
            int y=kyPos(k,inLoc->nx+inLoc->halo.lt+inLoc->halo.rt,inLoc->ny+inLoc->halo.dn+inLoc->halo.up,inLoc->nf)-inLoc->halo.up+inLoc->ky0;
            int f=featureIndex(k,inLoc->nx+inLoc->halo.lt+inLoc->halo.rt,inLoc->ny+inLoc->halo.dn+inLoc->halo.up,inLoc->nf);
            pvdata_t a = inLayer->getLayerData()[k];
            
            if (x>=0 && x<inLoc->nxGlobal && y>=0 && y<inLoc->nyGlobal) {
               int kRestricted = kIndex(x,y,f,inLoc->nxGlobal,inLoc->nyGlobal,inLoc->nf);
               pvInfo().printf("Rank %d, kLocal(extended)=%d, kGlobal(restricted)=%2d, x=%2d, y=%2d, f=%2d, a=%f\n", r, k, kRestricted, x, y, f, a);
               pvdata_t correctValue = (pvdata_t) kRestricted+1.0f;
               if (a!=correctValue) {
                  status = PV_FAILURE;
                  pvErrorNoExit().printf("        Failure! Correct value is %f\n", correctValue);
               }
            }
         }
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
   }

   // Connection should be a 3x3 kernel with values 0 through 8 in the weights
   BaseConnection * baseConn = hc->getConnection(0);
   HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(conn->xPatchSize()==3 && conn->yPatchSize()==3 && conn->fPatchSize()==1), "Test failed.\n");
   int patchSize = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
   pvErrorIf(!(conn->numberOfAxonalArborLists()==1), "Test failed.\n");
   pvErrorIf(!(conn->getNumDataPatches()==1), "Test failed.\n");
   pvwdata_t * w = conn->get_wDataHead(0,0);
   for (int r=0; r<hc->getCommunicator()->commSize(); r++) {
      if (r==hc->columnId()) {
         pvInfo().printf("Rank %d, Weight values\n", r);
         for (int k=0; k<patchSize; k++) {
            pvInfo().printf("Rank %d, k=%2d, w=%f\n", r, k, w[k]);
            if (w[k]!=(pvdata_t) k) {
               status = PV_FAILURE;
               pvErrorNoExit().printf("        Failure! Correct value is %f\n", (pvdata_t) k);
            }
         }
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
   }
   for (int k=0; k<patchSize; k++) {
      pvErrorIf(!(w[k]==(pvdata_t) k), "Test failed.\n");
   }

   // Finally, output layer should be 2x2 with values [13 23; 43 53].
   HyPerLayer * outLayer = hc->getLayer(1);
   const PVLayerLoc * outLoc = outLayer->getLayerLoc();
   pvErrorIf(!(outLoc->nxGlobal==2 && outLoc->nyGlobal==2 && outLoc->nf==1), "Test failed.\n");
   assert(outLoc->halo.lt==0 &&
          outLoc->halo.rt==0 &&
          outLoc->halo.dn==0 &&
          outLoc->halo.up==0 &&
          outLayer->getNumGlobalExtended()==4);
   const pvdata_t correct[4] = {13.0f, 23.0f, 43.0f, 53.0f};
   
   for (int r=0; r<hc->getCommunicator()->commSize(); r++) {
      if (r==hc->columnId()) {
         pvInfo().printf("Rank %d, Output layer V\n",r);
         for (int k=0; k<outLayer->getNumNeurons(); k++) {
            int x=kxPos(k,outLoc->nx,outLoc->ny,outLoc->nf)+outLoc->kx0;
            int y=kyPos(k,outLoc->nx,outLoc->ny,outLoc->nf)+outLoc->ky0;
            int f=featureIndex(k,outLoc->nxGlobal,outLoc->nyGlobal,outLoc->nf);
            pvdata_t V = outLayer->getV()[k];
            
            if (x>=0 && x<outLoc->nxGlobal && y>=0 && y<outLoc->nyGlobal) {
               int kRestricted = kIndex(x,y,f,outLoc->nxGlobal,outLoc->nyGlobal,outLoc->nf);
               pvInfo().printf("Rank %d, kLocal=%d, kGlobal=%2d, x=%2d, y=%2d, f=%2d, V=%f\n", r, k, kRestricted, x, y, f, V);
               if (V!=correct[kRestricted]) {
                  status = PV_FAILURE;
                  pvErrorNoExit().printf("        Failure! Correct value is %f\n", correct[kRestricted]);
               }
            }
         }
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
   }

   return status;
}
