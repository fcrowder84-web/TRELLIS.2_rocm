"""
UV-space rasterizer using a custom HIP kernel via torch.utils.cpp_extension.

Custom GPU kernel that:
  1. For each texel, finds which UV triangle covers it (AABB culling + point-in-triangle)
  2. Computes barycentric coordinates with float-precision UVs
  3. Interpolates the 3D vertex position

The nvdiffrast ROCm port has a wave32 shuffle bug that produces ~48% coverage
for fullscreen triangles, so this kernel replaces it for UV-space texture baking.

Compiled lazily on first use via torch.utils.cpp_extension.load().
"""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

_MODULE = None
_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "uv_rasterize_kernel.hip")


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load(
            name="uv_rasterize_kernel",
            sources=[_KERNEL_PATH],
            verbose=False,
            extra_cuda_cflags=["-w", "-O3"],
        )
    return _MODULE


@torch.no_grad()
def uv_rasterize(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    vertices: torch.Tensor,
    texture_size: int,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize UV triangles into texture space on GPU.

    Args:
        uvs:       (num_verts, 2) float32 on GPU — UV coordinates in [0, 1]
        faces:     (num_faces, 3) int32 on GPU — vertex indices
        vertices:  (num_verts, 3) float32 on GPU — 3D positions
        texture_size: Output texture resolution (square)
        verbose:   Print timing/coverage info

    Returns:
        face_ids: (TS, TS) int32 — face index + 1 per texel (0 = uncovered)
        pos:      (TS, TS, 3) float32 — interpolated 3D position per texel
    """
    num_faces = faces.shape[0]
    TS = texture_size
    scale = float(TS - 1)

    if verbose:
        print(f"[uv_rasterize] {num_faces} faces, TS={TS}")

    # Build per-triangle packed data (20 floats per tri):
    # [minU, minV, maxU, maxV, u0, v0, u1, v1, u2, v2, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, face_id]
    f = faces.long()
    uv0 = uvs[f[:, 0]] * scale
    uv1 = uvs[f[:, 1]] * scale
    uv2 = uvs[f[:, 2]] * scale
    p0 = vertices[f[:, 0]]
    p1 = vertices[f[:, 1]]
    p2 = vertices[f[:, 2]]

    minU = torch.minimum(torch.minimum(uv0[:, 0], uv1[:, 0]), uv2[:, 0])
    minV = torch.minimum(torch.minimum(uv0[:, 1], uv1[:, 1]), uv2[:, 1])
    maxU = torch.maximum(torch.maximum(uv0[:, 0], uv1[:, 0]), uv2[:, 0])
    maxV = torch.maximum(torch.maximum(uv0[:, 1], uv1[:, 1]), uv2[:, 1])

    tri_data = torch.stack([
        minU, minV, maxU, maxV,
        uv0[:, 0], uv0[:, 1],
        uv1[:, 0], uv1[:, 1],
        uv2[:, 0], uv2[:, 1],
        p0[:, 0], p0[:, 1], p0[:, 2],
        p1[:, 0], p1[:, 1], p1[:, 2],
        p2[:, 0], p2[:, 1], p2[:, 2],
        torch.arange(num_faces, device=uvs.device, dtype=torch.float32),
    ], dim=1).contiguous()

    # Build per-row bins on GPU: for each texture row, list triangle indices
    # whose V range overlaps that row.
    minRow = torch.clamp(minV.floor().long(), 0, TS - 1)
    maxRow = torch.clamp(maxV.ceil().long(), 0, TS - 1)

    # Build bin arrays on CPU (one pass over rows)
    # Each row: [count, idx0, idx1, ...] padded to max_bins_per_row + 1
    minRow_cpu = minRow.cpu().numpy()
    maxRow_cpu = maxRow.cpu().numpy()

    # Count per row
    row_counts = np.zeros(TS, dtype=np.int32)
    for i in range(num_faces):
        for r in range(minRow_cpu[i], maxRow_cpu[i] + 1):
            row_counts[r] += 1

    max_bins_per_row = int(row_counts.max()) if num_faces > 0 else 1
    bin_stride = max_bins_per_row + 1  # [count, idx0, idx1, ...]

    if verbose:
        print(f"[uv_rasterize] max_tris_per_row={max_bins_per_row}, bin_stride={bin_stride}")

    # Fill bins
    row_bins_np = np.zeros((TS, bin_stride), dtype=np.int32)
    row_cursors = np.zeros(TS, dtype=np.int32)
    for i in range(num_faces):
        for r in range(minRow_cpu[i], maxRow_cpu[i] + 1):
            row_bins_np[r, 1 + row_cursors[r]] = i
            row_cursors[r] += 1
    for r in range(TS):
        row_bins_np[r, 0] = row_cursors[r]

    row_bins = torch.from_numpy(row_bins_np).cuda()

    face_ids = torch.zeros(TS, TS, dtype=torch.int32, device=uvs.device)
    out_pos = torch.zeros(TS, TS, 3, dtype=torch.float32, device=uvs.device)

    mod = _get_module()
    mod.uv_rasterize_launch(
        tri_data, row_bins,
        num_faces, TS, bin_stride,
        face_ids, out_pos
    )

    if verbose:
        cov = (face_ids > 0).sum().item()
        print(f"[uv_rasterize] Coverage: {cov}/{TS*TS} ({cov/(TS*TS)*100:.1f}%)")

    return face_ids, out_pos
