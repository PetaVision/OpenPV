#ifndef PVDATA_HPP_
#define PVDATA_HPP_

#include <vector>

namespace PV {

typedef std::vector<std::pair<float, int>> SparseVector;
typedef std::vector<float> DenseVector;


class BasePack {
   protected:
      int mNB, mNY, mNX, mNF;

      // Bounds checking / index conversion methods
      int index(int y, int x, int f) { return y * mNX * mNF + x * mNF + f; }
      int elements() { return mNY * mNX * mNF; }
      bool batchOk(int b) { return b >= 0 && b < mNB; }
      bool indexOk(int i) { return i >= 0 && i < elements(); }

   public:
      // Returns the value at the specified location. OOB returns 0.0f
      virtual float get(int batch, int y, int x, int f) = 0;
      // Returns true if successful, false for OOB or other error
      virtual bool set(int batch, int y, int x, int f, float v) = 0;
      virtual bool set(int batch, SparseVector *v) = 0;
      virtual bool set(int batch, DenseVector *v) = 0;
      // Returns an empty vector on error
      virtual SparseVector asSparse(int batch) = 0;
      virtual DenseVector asDense(int batch) = 0;
   
      int getNB() { return mNB; }
      int getNY() { return mNY; }
      int getNX() { return mNX; }
      int getNF() { return mNF; }
};

class SparsePack : BasePack {

   public:
      float get(int batch, int y, int x, int f) override;
      bool set(int batch, int y, int x, int f, float v) override;
      bool set(int batch, SparseVector *v) override;
      bool set(int batch, DenseVector *v) override;
      SparseVector asSparse(int batch) override;
      DenseVector asDense(int batch) override;
   private:
      std::vector<SparseVector*> mData;
};

class DensePack : BasePack {

   public:
      float get(int batch, int y, int x, int f) override;
      bool set(int batch, int y, int x, int f, float v) override;
      bool set(int batch, SparseVector *v) override;
      bool set(int batch, DenseVector *v) override;
      SparseVector asSparse(int batch) override;
      DenseVector asDense(int batch) override;
   private:
      std::vector<DenseVector*> mData;
};

} /* namespace PV */
 
#endif /* PVDATA_HPP_ */
