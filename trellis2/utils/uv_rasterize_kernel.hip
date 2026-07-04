// uv_rasterize_kernel.hip — GPU UV-space rasterizer for TRELLIS.2 ROCm
// Custom HIP kernel replacing nvdiffrast for UV-space texture baking.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void uv_rasterize_kernel(
    const float* __restrict__ tri_data,
    const int*   __restrict__ row_bins,
    int   num_tris,
    int   texture_size,
    int   bin_stride,
    int*   __restrict__ face_ids,
    float* __restrict__ out_pos
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= texture_size || ty >= texture_size) return;

    float px = (float)tx;
    float py = (float)ty;

    const int* row = &row_bins[ty * bin_stride];
    int row_count = row[0];

    int best_face = 0;
    float best_x = 0.f, best_y = 0.f, best_z = 0.f;

    for (int i = 0; i < row_count; i++) {
        int tri_idx = row[1 + i];
        const float* tri = &tri_data[tri_idx * 20];

        float minU = tri[0], minV = tri[1], maxU = tri[2], maxV = tri[3];
        if (px < minU || px > maxU || py < minV || py > maxV)
            continue;

        float u0 = tri[4], v0 = tri[5];
        float u1 = tri[6], v1 = tri[7];
        float u2 = tri[8], v2 = tri[9];

        float denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
        if (fabsf(denom) < 1e-6f) continue;
        float inv_denom = 1.0f / denom;

        float w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv_denom;
        float w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv_denom;
        float w2 = 1.0f - w0 - w1;

        float eps = -0.001f;
        if (w0 < eps || w1 < eps || w2 < eps)
            continue;

        float p0x = tri[10], p0y = tri[11], p0z = tri[12];
        float p1x = tri[13], p1y = tri[14], p1z = tri[15];
        float p2x = tri[16], p2y = tri[17], p2z = tri[18];

        best_x = w0 * p0x + w1 * p1x + w2 * p2x;
        best_y = w0 * p0y + w1 * p1y + w2 * p2y;
        best_z = w0 * p0z + w1 * p1z + w2 * p2z;
        best_face = (int)tri[19] + 1;

        break;
    }

    int idx = ty * texture_size + tx;
    face_ids[idx] = best_face;
    out_pos[idx * 3 + 0] = best_x;
    out_pos[idx * 3 + 1] = best_y;
    out_pos[idx * 3 + 2] = best_z;
}

void uv_rasterize_launch(
    torch::Tensor tri_data,
    torch::Tensor row_bins,
    int num_tris,
    int texture_size,
    int bin_stride,
    torch::Tensor face_ids,
    torch::Tensor out_pos
) {
    int TS = texture_size;
    dim3 block(16, 16);
    dim3 grid((TS + block.x - 1) / block.x, (TS + block.y - 1) / block.y);

    hipStream_t stream = 0; // default stream

    uv_rasterize_kernel<<<grid, block, 0, stream>>>(
        tri_data.data_ptr<float>(),
        row_bins.data_ptr<int>(),
        num_tris,
        TS,
        bin_stride,
        face_ids.data_ptr<int>(),
        out_pos.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uv_rasterize_launch", &uv_rasterize_launch, "UV-space rasterizer kernel");
}
