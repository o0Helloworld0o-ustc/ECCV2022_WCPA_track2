import cv2
import numpy as np

from Sim3DR import RenderPipeline



def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

cfg = {
    "intensity_ambient": 0.3,
    "color_ambient": (1, 1, 1),
    "intensity_directional": 0.6,
    "color_directional": (1, 1, 1),
    "intensity_specular": 0.1,
    "specular_exp": 5,
    "light_pos": (0, 0, 5),
    "view_pos": (0, 0, 5),
}
render_app = RenderPipeline(**cfg)



class QuickRenderer(object):

    def __init__(self, img_w, img_h, M_proj, tris, alpha=0.7):
        self.img_w, self.img_h = img_w, img_h
        self.alpha = alpha

        self.M_proj = M_proj
        self.focal = M_proj[0, 0] * (0.5 * img_w)

        tris[:, [0, 1]] = tris[:, [1, 0]]
        self.tris = np.ascontiguousarray(tris.astype(np.int32))

        self.M1 = np.array([
            [img_w/2,       0, 0, 0],
            [      0, img_h/2, 0, 0],
            [      0,       0, 1, 0],
            [img_w/2, img_h/2, 0, 1]
        ])



    def __call__(self, verts3d, R_t, overlap=None):
        ones = np.ones([verts3d.shape[0], 1])
        verts_homo = np.concatenate([verts3d, ones], axis=1)

        verts = verts_homo @ R_t @ self.M_proj @ self.M1
        w_ = verts[:, [3]]
        verts = verts / w_

        # image space: →+x，↓+y
        points2d = verts[:, :2]
        points2d[:, 1] = self.img_h - points2d[:, 1]

        verts_temp = np.concatenate([points2d, w_], axis=1)

        tz = R_t[3, 2]
        scale = self.focal / tz
        verts_temp[:, 2] *= scale

        verts_temp = _to_ctype(verts_temp.astype(np.float32))

        if overlap is None:
            overlap = np.zeros([self.img_h, self.img_w, 3], dtype=np.uint8)

        overlap_copy = overlap.copy()
        overlap = render_app(verts_temp, self.tris, overlap)

        img_render = cv2.addWeighted(overlap_copy, 1 - self.alpha, overlap, self.alpha, 0)
        return img_render

