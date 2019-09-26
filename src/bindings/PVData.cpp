#include "bindings/PVData.hpp"

namespace PV {

float SparsePack::get(int batch, int y, int x, int f) {
   int i = index(y, x, f);

   if (!batchOk(batch) || !indexOk(i)) {
      return 0.0f;
   }
   
   SparseVector *vec = mData[batch];
   int L = 0;
   int R = vec->size() - 1;

   // Nothing is active
   if (vec->empty()) {
      return 0.0f;
   }
   vec = mData[batch];

   // Single element, only one valid index
   if (L == R) {
      if (i == (*vec)[0].second) {
         return (*vec)[0].first;
      }
      return 0.0f;
   }

   // Binary search for index in sparse list
   do {
      int m = (L + R) >> 1; 
      if ((*vec)[m].second < i) {
         L = m + 1;
      }
      else if ((*vec)[m].second > i) {
         R = m - 1;
      }
      else {
         return (*vec)[m].first;
      }
   } while (L <= R);

   return 0.0f; 
}

bool SparsePack::set(int batch, int y, int x, int f, float v) {
   int i = index(y, x, f);

    if (!batchOk(batch) || !indexOk(i)) {
      return false;
   }

   SparseVector *vec = mData[batch];
   int L = 0;
   int R = vec->size() - 1;

   // Nothing is active
   if (vec->empty()) {
      vec->push_back({v, i});
      return true;
   }

   // Single element, only one valid index
   if (L == R) {
      if (i > (*vec)[0].second) {
         vec->push_back({v, i});
         return true;
      }
      if (i < (*vec)[0].second) {
         vec->insert(vec->begin(), {v, i});
         return true;
      }
      (*vec)[0].first = v;
      return true;
   }

   // Binary search for index in sparse list
   int m;
   do {
      m = (L + R) >> 1; 
      if ((*vec)[m].second < i) {
         L = m + 1;
      }
      else if ((*vec)[m].second > i) {
         R = m - 1;
      }
      else {
         (*vec)[m].first = v;
         return true;
      }
   } while (L <= R);

   vec->insert(vec->begin()+m, {v, i}); 
   return true;
}

bool SparsePack::set(int batch, SparseVector *v) {
   if (!batchOk(batch)) {
      return false;
   }
   mData[batch] = v;
   return true;
}

bool SparsePack::set(int batch, DenseVector *v) {
   int n = elements();
   if(n != v->size() || !batchOk(batch)) {
      return false;
   } 

   SparseVector *sv = new SparseVector();

   for (int i = 0; i < n; i++) {
      if ((*v)[i] != 0) {
         sv->push_back({(*v)[i], i});
      }
   }

   if (mData[batch] != nullptr) {
      delete mData[batch];
   }
   mData[batch] = sv;

   return true;
}

SparseVector SparsePack::asSparse(int batch) {
   if (!batchOk(batch)) {
      return SparseVector();
   }
   return *mData[batch];
}

DenseVector SparsePack::asDense(int batch) {
   DenseVector dv;
   if (!batchOk(batch)) {
      return dv;
   }
   dv.resize(elements());
   for (auto p : *mData[batch]) {
      dv[p.second] = p.first;
   }
   return dv;
}

float DensePack::get(int batch, int y, int x, int f) {
   int i = index(y, x, f);
   if (!batchOk(batch) || !indexOk(i)) {
      return 0.0f;
   }
   return (*mData[batch])[i];
}

bool DensePack::set(int batch, int y, int x, int f, float v) {
   int i = index(y, x, f);
   if (!batchOk(batch) || !indexOk(i)) {
      return false;
   }
   (*mData[batch])[i] = v;
   return true;
}

bool DensePack::set(int batch, SparseVector *v) {
   if (!batchOk(batch)) {
      return false;
   }
   DenseVector *dv = new DenseVector();
   dv->resize(elements());
   for (auto p : *v) {
      (*dv)[p.second] = p.first;
   }
   if (mData[batch] != nullptr) {
      delete mData[batch];
   }
   mData[batch] = dv;
   return true;
}

bool DensePack::set(int batch, DenseVector *v) {
   if (!batchOk(batch)) {
      return false;
   }
   mData[batch] = v;
   return true;
}

SparseVector DensePack::asSparse(int batch) {
   SparseVector sv;
   if (!batchOk(batch)) {
      return sv;
   }
   for (int i = 0; i < elements(); i++) {
      if ((*mData[batch])[i] != 0.0f) {
         sv.push_back({(*mData[batch])[i], i});
      }
   }
   return sv;
}

DenseVector DensePack::asDense(int batch) {
   if (!batchOk(batch)) {
      return DenseVector();
   }
   return *mData[batch];
}

}; /* namespace PV */
