#ifndef PVDATA_HPP_
#define PVDATA_HPP_

#include <vector>

namespace PV {

typedef std::vector<std::pair<int, float>> SparseVector;
typedef std::vector<float> DenseVector;


class DataPack {
   protected:
      int mNB, mNY, mNX, mNF;

      // Bounds checking / index conversion methods
      int index(int y, int x, int f) const { return y * mNX * mNF + x * mNF + f; }
      bool batchOk(int b) const { return b >= 0 && b < mNB; }
      bool indexOk(int i) const { return i >= 0 && i < elements(); }

   public:
      // Returns the value at the specified location. OOB returns 0.0f
      virtual float get(int batch, int y, int x, int f) = 0;
      // Returns true if successful, false for OOB or other error
      virtual bool set(int batch, int y, int x, int f, float v) = 0;
      virtual bool set(int batch, SparseVector *v) = 0;
      virtual bool set(int batch, DenseVector *v) = 0;
      // Returns an empty vector on error
      virtual SparseVector asSparse(int batch) const = 0;
      virtual DenseVector asDense(int batch) const = 0;

      virtual const char *format() const = 0;
   
      int elements() const { return mNY * mNX * mNF; }
      int getNB() const { return mNB; }
      int getNY() const { return mNY; }
      int getNX() const { return mNX; }
      int getNF() const { return mNF; }
};

class SparsePack : public DataPack {
   public:
      SparsePack(int nbatch, int ny, int nx, int nf);
      ~SparsePack();
      float get(int batch, int y, int x, int f) override;
      bool set(int batch, int y, int x, int f, float v) override;
      bool set(int batch, SparseVector *v) override;
      bool set(int batch, DenseVector *v) override;
      SparseVector asSparse(int batch) const override;
      DenseVector asDense(int batch) const override;
      const char *format() const override { return "sparse"; }
   private:
      std::vector<SparseVector> mData;
};

class DensePack : public DataPack {
   public:
      DensePack(int nbatch, int ny, int nx, int nf);
      ~DensePack();
      float get(int batch, int y, int x, int f) override;
      bool set(int batch, int y, int x, int f, float v) override;
      bool set(int batch, SparseVector *v) override;
      bool set(int batch, DenseVector *v) override;
      SparseVector asSparse(int batch) const override;
      DenseVector asDense(int batch) const override;
      const char *format() const override { return "dense"; }
   private:
      std::vector<DenseVector> mData;
};

} /* namespace PV */
 
#endif /* PVDATA_HPP_ */
