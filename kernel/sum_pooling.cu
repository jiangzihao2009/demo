#include <cuda.h>
#include <cuda_runtime.h>

#include <random>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;
constexpr int kEmbPerBlock = 8;

__forceinline__ __device__ unsigned lane_id() {
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id() {
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

struct Offset {
  int start;
  int num;
  int feature_idx;
  int batch_idx;
};

void PrintOffset(const std::vector<Offset>& offset) {
  for (int i = 0; i < offset.size(); i++) {
    auto& f = offset[i];
    cout << "offset[" << i << "] start: " << f.start << ", num: " << f.num
         << ", feature idx: " << f.feature_idx << ", batch idx: " << f.batch_idx
         << endl;
  }
}

void GenEmbedding(const int embedding_len, const int emb_cnt, std::vector<float>* emb) {
  size_t total_cnt = embedding_len * emb_cnt;
  emb->resize(total_cnt);
  for (size_t i = 0; i  < total_cnt; i++) {
    (*emb)[i] = (rand() % 220 - 110) * 0.01;
  }
}

void PrintEmbedding(const std::vector<float> & emb, const int embedding_len, const int emb_cnt) {
  for (int i = 0; i < emb_cnt; i++) {
    cout << "idx " << i << ": ";
    for (int j = 0; j < embedding_len; j++) {
      cout << emb[i * embedding_len + j] << ',';
    }
    cout << endl;
  }
}

// Output index, batch index pair.
using IndexPair = std::pair<uint16_t, uint16_t>;
bool CalculateIndex(const std::vector<Offset>& offset, std::vector<IndexPair>* index) {
  if (index == nullptr) {
    return false;
  }
  int start_block = 0, end_block = 0, start_idx = 0, end_idx = 0;
  int total_cnt = 0;
  for (int i = 0; i < offset.size(); i++) {
    auto& o = offset[i];
    start_idx = o.start;
    end_idx = o.start + o.num;
    for (int j = start_idx; j < end_idx; j++) {
      (*index)[j].first = o.feature_idx;
      (*index)[j].second = o.batch_idx;
    }
  }
  return true;
}

__global__ void SumPoolingByBatch(
    const int embedding_len, const IndexPair* index, const float* input, float** output) {
  const int32_t emb_idx = blockIdx.x * kEmbPerBlock + threadIdx.y;
  const size_t start_offset = emb_idx * embedding_len;
  const IndexPair info = index[emb_idx];
  atomicAdd(&output[info.first][info.second * embedding_len + threadIdx.x], input[start_offset + threadIdx.x]);
}

__global__ void TestLane() {
  printf("%d, %d, %d\n", threadIdx.x, threadIdx.y, lane_id());
}

int main() {
  int feature_cnt = 3;
  int batch_cnt = 2;
  int embedding_len = 8;

  int TPB = std::min<int>(32, embedding_len * 4);
  int total_cnt = 0;
  std::vector<Offset> offset;
  for (int i = 0; i < feature_cnt; i++) {
    for (int j = 0; j < batch_cnt; j++) {
      int item_cnt = rand() % 4 + 1;
      auto& last = offset.emplace_back();
      last.num = item_cnt;
      last.start = total_cnt;
      last.feature_idx = i;
      last.batch_idx = j;
      total_cnt += item_cnt;
    }
  }
  PrintOffset(offset);
  int block_cnt = (total_cnt + kEmbPerBlock - 1) / kEmbPerBlock;
  std::vector<IndexPair> index(total_cnt);
  CalculateIndex(offset, &index);

  IndexPair* d_index;
  cudaMalloc(&d_index, total_cnt * sizeof(IndexPair));
  cudaMemcpy(&d_index, index.data(), index.size() * sizeof(IndexPair),
             cudaMemcpyHostToDevice);
  // Allocate two dimension array.
  float** d_output;
  std::vector<float*> d_help(feature_cnt);
  cudaMalloc(&d_output, feature_cnt * sizeof(float*));
  for (int i = 0; i < feature_cnt; i++) {
    cudaMalloc(&d_help[i], embedding_len * batch_cnt * sizeof(float));
    cudaMemset(&d_help[i], 0, embedding_len * batch_cnt * sizeof(float));
  }
  cudaMemcpy(d_output, d_help.data(), feature_cnt * sizeof(float*), cudaMemcpyHostToDevice);
  cout << "total count: " << total_cnt
       << ", index count: " << index.size() << endl;
  std::vector<float> emb;
  GenEmbedding(embedding_len, total_cnt, &emb);
  PrintEmbedding(emb, embedding_len, total_cnt);
  float* d_input;
  cudaMalloc(&d_input, emb.size() * sizeof(float));
  cudaMemcpy(d_input, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(embedding_len, kEmbPerBlock);
  SumPoolingByBatch<<<block_cnt, threads>>>(embedding_len, d_index, d_input, d_output);
  std::vector<float> o_val(embedding_len * batch_cnt * sizeof(float));
  for (int i =0; i < feature_cnt; i++) {
    cudaMemcpy(o_val.data(), d_help[i],
      embedding_len * batch_cnt * sizeof(float), cudaMemcpyDeviceToHost);
    PrintEmbedding(o_val, embedding_len, batch_cnt);
  }
  // dim3 threads(8, 4);
  // TestLane<<<1, threads>>>();
  cudaDeviceSynchronize();
  return 0;
}