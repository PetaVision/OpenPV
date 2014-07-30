//
//  SoundReconLayer.cpp
//  
//
//  Created by Brohit's Mac 5054128794 on 7/24/14.
//
//

#include "SoundReconLayer.h"


namespace PV {
    
    SoundReconLayer::SoundReconLayer() {
        initialize_base();
    }
    
    SoundReconLayer::SoundReconLayer(const char * name, HyPerCol * hc) {
        initialize_base();
        initialize(name, hc);
    }  // end SoundReconLayer::SoundReconLayer(const char *, HyPerCol *)
    
    SoundReconLayer::~SoundReconLayer() {
            }
    
    int SoundReconLayer::initialize_base() {
        
        bufferLevel = 0;
        return PV_SUCCESS;
    }
    
    int SoundReconLayer::resetGSynBuffers(double timef, double dt) {
        
        return PV_SUCCESS;
    }
    
    int SoundReconLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
                      pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking,
                                       unsigned int * active_indices, unsigned int * num_active) {
        int nx = getLayerLoc()->nx;
        int ny = getLayerLoc()->ny;
        int nf = getLayerLoc()->nf;
        
        for (int i = 0; i < nx; i++) {
            int vx = i;
            int gx = i;
            for (int j = 0; j < nf; j ++) {
                int vf = j;
                int gf;
                if (vf + bufferLevel < nf) {
                    gf = bufferLevel + j;
                }
                
                else {
                    gf = bufferLevel + j - nf;
                }
                
                int vindex = kIndex(vx, 0, vf, nx, ny, nf);
                int gindex = kIndex(gx, 0, gf, nx, ny, nf);
                
                V[vindex] = GSyn[0][gindex];
            }
        }
        return PV_SUCCESS;
    }
    
    int SoundReconLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
        
        //Check if we need to update based on connection's channel
        if(conn->getChannel() == CHANNEL_NOUPDATE){
            return PV_SUCCESS;
        }
        
        recvsyn_timer->start();
        
        assert(arborID >= 0);
        const int numExtended = activity->numItems;
        
#ifdef DEBUG_OUTPUT
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
        printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
        fflush(stdout);
#endif // DEBUG_OUTPUT
        
        float dt_factor = getConvertToRateDeltaTimeFactor(conn);
        
        
        //Clear all thread gsyn buffer
        /*
        if(thread_gSyn){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for(int i = 0; i < parent->getNumThreads() * getNumNeurons(); i++){
                thread_gSyn[0][i] = 0;
            }
        }
        
        //If we're using thread_gSyn, set this here
        pvdata_t * gSynPatchHead;
         
#ifdef PV_USE_OPENMP_THREADS
        if(thread_gSyn){
            int ti = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
        }
        else{
            gSynPatchHead = this->getChannel(conn->getChannel());
        }
#else
        gSynPatchHead = this->getChannel(conn->getChannel());
#endif
         
         */
        
        pvdata_t * gSynPatchHead;
        gSynPatchHead = this->getChannel(conn->getChannel());
        //taken out from comments when not using threads
        for (int j = 0; j < getLayerLoc()->nx; j++) {
            
            gSynPatchHead[bufferLevel + j *  getLayerLoc()->nf] = 0;
            
        }
        
        //clears memory slot to be written to
        
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
        for (int kPre = 0; kPre < numExtended; kPre++) {
            bool inWindow;
            //Post layer recieves synaptic input
            //Only with respect to post layer
            const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
            const PVLayerLoc * postLoc = this->getLayerLoc();
            int kPost = layerIndexExt(kPre, preLoc, postLoc);
            inWindow = inWindowExt(arborID, kPost);
            if(!inWindow) continue;
            
            float a = activity->data[kPre] * dt_factor;
            // Activity < 0 is used by generative models --pete
            if (a == 0.0f) continue;
            
            PVPatch * weights = conn->getWeights(kPre, arborID);
            
            // WARNING - assumes weight and GSyn patches from task same size
            //         - assumes patch stride sf is 1
            
            int nx  = weights->nx;
            int ny  = weights->ny;
            int nf  = conn->fPatchSize();
            int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
            int syw = conn->yPatchStride();                   // stride in patch
            
            size_t gSynPatchStartIndex = conn->getGSynPatchStart(kPre, arborID) + bufferLevel;
            pvdata_t * gSynPatchStart = gSynPatchHead + gSynPatchStartIndex;
            // GTK: gSynPatchStart redefined as offset from start of gSyn buffer
            pvwdata_t * data = conn->get_wData(arborID,kPre);
            uint4 * rngPtr = conn->getRandState(kPre);
            
            assert (ny == 1); //for loop applies only when ny = 1
            
            for (int j = 0; j < nx; j++) {
                gSynPatchStart[j * nf] += data[j * nf] * a;
            }
            
            if (bufferLevel < nf - 1) {
                bufferLevel++; }
            else {
                bufferLevel = 0;
            }
            
        }
        
        
#ifdef PV_USE_OPENMP_THREADS
        //Accumulate back into gSyn
        if(thread_gSyn){
            pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
            //Looping over neurons first to be thread safe
#pragma omp parallel for
            for(int ni = 0; ni < getNumNeurons(); ni++){
                for(int ti = 0; ti < parent->getNumThreads(); ti++){
                    gSynPatchHead[ni] += thread_gSyn[ti][ni];
                }
            }
        }
#endif
        
        recvsyn_timer->stop();
        
        return PV_SUCCESS;
    }
    
}  // end namespace PV

