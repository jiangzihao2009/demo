#pragma once
#include <immintrin.h>

inline void HalfToFloat(float *target, const uint16_t *source, size_t len) {
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    __m128i source_val = _mm_loadu_si128((__m128i *)(source + i));
    __m256 target_val = _mm256_cvtph_ps(source_val);
    _mm256_storeu_ps(target + i, target_val);
  }
  for (; i < len; ++i) {
    target[i] = _cvtsh_ss(source[i]);
  }
}

inline void FloatToHalf(uint16_t *target, const float *source, size_t len) {
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    __m256 source_val = _mm256_loadu_ps(source + i);
    __m128i target_val = _mm256_cvtps_ph(
        source_val, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i *)(target + i), target_val);
  }
  for (; i < len; ++i) {
    target[i] = _cvtss_sh(source[i], 0);
  }
}
