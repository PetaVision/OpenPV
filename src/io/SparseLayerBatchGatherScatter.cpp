#include "SparseLayerBatchGatherScatter.hpp"

namespace PV {

SparseLayerBatchGatherScatter::SparseLayerBatchGatherScatter(
         std::shared_ptr<MPIBlock const> mpiBlock,
         PVLayerLoc const &layerLoc,
         int rootProcessRank,
         bool localExtended,
         bool rootExtended) {

   mMPIBlock = mpiBlock;
   mLayerLoc = layerLoc;
   mRootProcessRank = rootProcessRank;
   if (!localExtended) {
      mLayerLoc.halo.lt = 0;
      mLayerLoc.halo.rt = 0;
      mLayerLoc.halo.dn = 0;
      mLayerLoc.halo.up = 0;
   }
   mRootExtended = rootExtended;
}

void SparseLayerBatchGatherScatter::gather(
      int mpiBatchIndex,
      SparseList<float> *rootSparseList,
      SparseList<float> const *localSparseList) {

   if (mMPIBlock->getRank() == mRootProcessRank) {
      int nxResGlobal = mLayerLoc.nx * mMPIBlock->getNumColumns();
      int nyResGlobal = mLayerLoc.ny * mMPIBlock->getNumRows();
      int nxExtGlobal = nxResGlobal + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      int nyExtGlobal = nyResGlobal + mLayerLoc.halo.dn + mLayerLoc.halo.up;
      rootSparseList->reset(nxExtGlobal, nyExtGlobal, mLayerLoc.nf);
      for (int recvRow = mMPIBlock->getNumRows() - 1; recvRow >= 0; --recvRow) {
         for (int recvColumn = mMPIBlock->getNumColumns() - 1; recvColumn >= 0; --recvColumn) {
            int recvRank = mMPIBlock->calcRankFromRowColBatch(recvRow, recvColumn, mpiBatchIndex);
            SparseList<float> recvList;
            if (recvRank != mRootProcessRank) {
               int numToRecv = 0;
               MPI_Recv(
                     &numToRecv, 1, MPI_INT, recvRank, 35, mMPIBlock->getComm(), MPI_STATUS_IGNORE);
               std::vector<SparseList<float>::Entry> recvBuffer;
               if (numToRecv > 0) {
                  recvBuffer.resize(numToRecv);
                  MPI_Recv(
                        recvBuffer.data(),
                        numToRecv * static_cast<int>(sizeof(SparseList<float>::Entry)),
                        MPI_BYTE,
                        recvRank,
                        36,
                        mMPIBlock->getComm(),
                        MPI_STATUS_IGNORE);
               }
               recvList.reset(
                     localSparseList->getWidth(),
                     localSparseList->getHeight(),
                     localSparseList->getFeatures());
               recvList.set(recvBuffer);
               recvList.grow(
                     rootSparseList->getWidth(),
                     rootSparseList->getHeight(),
                     mLayerLoc.nx * recvColumn,
                     mLayerLoc.ny * recvRow);
            }
            else {
               recvList = *localSparseList;
               recvList.grow(
                     rootSparseList->getWidth(),
                     rootSparseList->getHeight(),
                     mLayerLoc.nx * recvColumn,
                     mLayerLoc.ny * recvRow);
            }
            // recvList.grow(nxExtGlobal, nyExtGlobal, mLayerLoc.halo.lt, mLayerLoc.halo.up);
            rootSparseList->merge(recvList);
         }
      }
      if (!mRootExtended) {
         rootSparseList->crop(nxResGlobal, nyResGlobal, mLayerLoc.halo.lt, mLayerLoc.halo.up);
      }
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      std::vector<SparseList<float>::Entry> contents = localSparseList->getContents();
      int numToSend = static_cast<int>(contents.size());
      MPI_Send(&numToSend, 1, MPI_INT, mRootProcessRank, 35, mMPIBlock->getComm());
      if (numToSend > 0) {
         MPI_Send(
               contents.data(),
               numToSend * static_cast<int>(sizeof(SparseList<float>::Entry)),
               MPI_BYTE,
               mRootProcessRank,
               36,
               mMPIBlock->getComm());
      }
   }
}

void SparseLayerBatchGatherScatter::scatter(
      int mpiBatchIndex,
      SparseList<float> const *rootSparseList,
      SparseList<float> *localSparseList) {
   
   int nxExtLocal = mLayerLoc.nx + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
   int nyExtLocal = mLayerLoc.ny + mLayerLoc.halo.dn + mLayerLoc.halo.up;
   localSparseList->reset(nxExtLocal, nyExtLocal, mLayerLoc.nf);
   if (mMPIBlock->getRank() == mRootProcessRank) {
      SparseList<float> rootCopy = *rootSparseList;
      int nxResGlobal = mLayerLoc.nx * mMPIBlock->getNumColumns();
      int nyResGlobal = mLayerLoc.ny * mMPIBlock->getNumRows();
      int nxExtGlobal = nxResGlobal + mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      int nyExtGlobal = nyResGlobal + mLayerLoc.halo.dn + mLayerLoc.halo.up;
      if (!mRootExtended) {
         rootCopy.grow(nxExtGlobal, nyExtGlobal, mLayerLoc.halo.lt, mLayerLoc.halo.up);
      }
      for (int sendRow = mMPIBlock->getNumRows() - 1; sendRow >= 0; --sendRow) {
         for (int sendColumn = mMPIBlock->getNumColumns() - 1; sendColumn >= 0; --sendColumn) {
            int sendRank = mMPIBlock->calcRankFromRowColBatch(sendRow, sendColumn, mpiBatchIndex);
            SparseList<float> sendList = rootCopy.extract(
                   sendColumn * mLayerLoc.nx, sendRow * mLayerLoc.ny, nxExtLocal, nyExtLocal);
            if (sendRank != mRootProcessRank) {
               std::vector<SparseList<float>::Entry> sendBuffer = sendList.getContents();
               int numToSend = static_cast<int>(sendBuffer.size());
               MPI_Send(&numToSend, 1, MPI_INT, sendRank, 33, mMPIBlock->getComm());
               MPI_Send(
                     sendBuffer.data(),
                     numToSend * static_cast<int>(sizeof(SparseList<float>::Entry)),
                     MPI_BYTE,
                     sendRank,
                     34,
                     mMPIBlock->getComm());
            }
            else {
               localSparseList->set(sendList.getContents());
            }
         }
      }
   }
   else if (mpiBatchIndex == mMPIBlock->getBatchIndex()) {
      int numToRecv = 0;
      MPI_Recv(
            &numToRecv, 1, MPI_INT, mRootProcessRank, 33, mMPIBlock->getComm(), MPI_STATUS_IGNORE);
      std::vector<SparseList<float>::Entry> recvBuffer(numToRecv);
      MPI_Recv(
            recvBuffer.data(),
            numToRecv * static_cast<int>(sizeof(SparseList<float>::Entry)),
            MPI_BYTE,
            mRootProcessRank,
            34,
            mMPIBlock->getComm(),
            MPI_STATUS_IGNORE);
      localSparseList->set(recvBuffer);
   }
}

} // namespace PV 
