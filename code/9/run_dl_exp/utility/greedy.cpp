#include <algorithm>
#include <cstring>
#include <numeric>

double *weight;

bool cmp(int32_t a, int32_t b)
{
    return weight[a] > weight[b];
}

void _get_top_k_by_greedy(double *in_array, int32_t num_batch, int32_t num_item, int32_t k, int32_t *out_array)
{
    int32_t *tmp = new int32_t[num_item];
    for (int32_t i = 0; i < num_batch; i++) {
        weight = in_array+num_item*i;
        std::iota(tmp, tmp+num_item, 0);
        std::sort(tmp, tmp+num_item, cmp);
        memcpy(out_array+k*i, tmp, sizeof(int32_t)*k);
    }
    delete[] tmp;
}
