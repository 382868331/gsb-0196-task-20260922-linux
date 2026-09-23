"""Tests for the visibility adjudication module.

Expected values come from independent reference math written directly in
this file (hand-computed projections, ray-plane intersections, manual
bilinear weights); the module under test is never used to derive them.
"""

import math
import unittest

from visibility import (
    DepthMap,
    PinholeCamera,
    Status,
    VisibilityError,
    classify_point,
    classify_points,
    points_visible_in_at_least,
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def make_camera(fx=1.0, fy=1.0, cx=0.0, cy=0.0,
                rotation=IDENTITY, translation=(0.0, 0.0, 0.0)):
    return PinholeCamera(fx, fy, cx, cy, rotation, translation)


def const_map(width, height, depth, disabled=None):
    return DepthMap(width, height, [depth] * (width * height), disabled)


class TestProjectionReference(unittest.TestCase):
    """Direct camera-projection reference, computed by hand."""

    def test_identity_camera_projection(self):
        # fx=fy=1, cx=cy=0: u = x/z, v = y/z.
        cam = make_camera()
        verdict = classify_point((7.5, 12.5, 5.0), cam, const_map(4, 4, 5.0))
        self.assertEqual(verdict.status, Status.VISIBLE)
        self.assertAlmostEqual(verdict.u, 1.5)
        self.assertAlmostEqual(verdict.v, 2.5)
        self.assertAlmostEqual(verdict.z, 5.0)
        self.assertAlmostEqual(verdict.depth, 5.0)

    def test_intrinsics_and_translation_reference(self):
        # Camera centre at world (1,2,3): p_cam = (4,5,6) - (1,2,3) = (3,3,3)
        # u = 2*3/3 + 0.5 = 2.5 ; v = 3*3/3 - 0.5 = 2.5
        cam = make_camera(fx=2.0, fy=3.0, cx=0.5, cy=-0.5,
                          translation=(-1.0, -2.0, -3.0))
        verdict = classify_point((4.0, 5.0, 6.0), cam, const_map(8, 8, 3.0))
        self.assertEqual(verdict.status, Status.VISIBLE)
        self.assertAlmostEqual(verdict.u, 2.5)
        self.assertAlmostEqual(verdict.v, 2.5)
        self.assertAlmostEqual(verdict.z, 3.0)

    def test_rotation_reference(self):
        # Rotation +90 deg about z: (x,y,z) -> (-y, x, z).
        # (1,0,5) -> (0,1,5); u = 0/5 = 0, v = 1/5 = 0.2.
        rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        cam = make_camera(rotation=rotation)
        verdict = classify_point((1.0, 0.0, 5.0), cam, const_map(4, 4, 5.0))
        self.assertEqual(verdict.status, Status.VISIBLE)
        self.assertAlmostEqual(verdict.u, 0.0)
        self.assertAlmostEqual(verdict.v, 0.2)


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.cam = make_camera()
        self.dmap = const_map(4, 4, 5.0)

    def project_at(self, u, v, z):
        """World point projecting to (u, v) with camera z under identity."""
        return (u * z, v * z, z)

    def test_visible(self):
        verdict = classify_point(self.project_at(1.5, 1.5, 5.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.VISIBLE)

    def test_occluded(self):
        verdict = classify_point(self.project_at(1.5, 1.5, 6.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.OCCLUDED)

    def test_in_front(self):
        verdict = classify_point(self.project_at(1.5, 1.5, 4.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.IN_FRONT)

    def test_tolerance_band_is_inclusive(self):
        depth = 5.0
        abs_tol, rel_tol = 0.1, 0.1
        tol = abs_tol + rel_tol * depth  # reference: 0.6
        for z, expected in ((depth + tol, Status.VISIBLE),
                            (depth + tol + 1e-9, Status.OCCLUDED),
                            (depth - tol, Status.VISIBLE),
                            (depth - tol - 1e-9, Status.IN_FRONT)):
            verdict = classify_point(self.project_at(1.5, 1.5, z),
                                     self.cam, self.dmap,
                                     abs_tol=abs_tol, rel_tol=rel_tol)
            self.assertEqual(verdict.status, expected,
                             f"z={z} should be {expected}")

    def test_behind_camera_is_unknown(self):
        for point in ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, -0.5)):
            verdict = classify_point(point, self.cam, self.dmap)
            self.assertEqual(verdict.status, Status.UNKNOWN)
            self.assertIsNone(verdict.depth)

    def test_projection_outside_map_is_unknown(self):
        # u = -0.5 < 0: no complete neighbourhood.
        verdict = classify_point(self.project_at(-0.5, 1.5, 5.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.UNKNOWN)
        # u = 3.9 with width 4: x0+1 = 4 is out of bounds.
        verdict = classify_point(self.project_at(3.9, 1.5, 5.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.UNKNOWN)

    def test_edge_missing_neighbourhood_is_unknown(self):
        # u exactly 3.0 in a width-4 map still lacks the x0+1 = 4 column.
        verdict = classify_point(self.project_at(3.0, 1.5, 5.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.UNKNOWN)
        # One pixel inward the neighbourhood is complete again.
        verdict = classify_point(self.project_at(2.0, 1.5, 5.0),
                                 self.cam, self.dmap)
        self.assertEqual(verdict.status, Status.VISIBLE)

    def test_bilinear_matches_manual_weights(self):
        # Row-major 2x2 map: d00=1, d10=2, d01=3, d11=4.
        dmap = DepthMap(2, 2, [1.0, 2.0, 3.0, 4.0])
        u, v = 0.25, 0.5
        expected = (1.0 * 0.75 * 0.5 + 2.0 * 0.25 * 0.5
                    + 3.0 * 0.75 * 0.5 + 4.0 * 0.25 * 0.5)  # = 2.25
        self.assertAlmostEqual(expected, 2.25)
        verdict = classify_point(self.project_at(u, v, expected),
                                 self.cam, dmap)
        self.assertEqual(verdict.status, Status.VISIBLE)
        self.assertAlmostEqual(verdict.depth, expected)

    def test_disabled_mask_forces_unknown(self):
        disabled = [False] * 16
        disabled[1 * 4 + 1] = True  # pixel (1, 1)
        dmap = const_map(4, 4, 5.0, disabled)
        # Neighbourhood of (1.5, 1.5) includes (1, 1).
        verdict = classify_point(self.project_at(1.5, 1.5, 5.0),
                                 self.cam, dmap)
        self.assertEqual(verdict.status, Status.UNKNOWN)
        self.assertIn("disabled", verdict.reason)
        # Neighbourhood of (0.5, 0.5) also includes (1, 1).
        verdict = classify_point(self.project_at(0.5, 0.5, 5.0),
                                 self.cam, dmap)
        self.assertEqual(verdict.status, Status.UNKNOWN)
        # Neighbourhood of (2.5, 2.5) avoids the disabled pixel.
        verdict = classify_point(self.project_at(2.5, 2.5, 5.0),
                                 self.cam, dmap)
        self.assertEqual(verdict.status, Status.VISIBLE)


class TestAnalyticPlaneReference(unittest.TestCase):
    """Analytic plane depth map via independent ray-plane intersection."""

    WIDTH = HEIGHT = 8
    FX = FY = 12.0
    CX = CY = (WIDTH - 1) / 2.0  # 3.5

    @staticmethod
    def plane_depth(px, py):
        """Depth of plane z = 5 + 0.3x + 0.2y along the pixel (px, py) ray.

        Ray: p(t) = t*(dx, dy, 1) with dx = (px - cx)/fx, dy = (py - cy)/fy.
        t = 5 + 0.3*t*dx + 0.2*t*dy  =>  t = 5 / (1 - 0.3*dx - 0.2*dy).
        """
        dx = (px - TestAnalyticPlaneReference.CX) / TestAnalyticPlaneReference.FX
        dy = (py - TestAnalyticPlaneReference.CY) / TestAnalyticPlaneReference.FY
        return 5.0 / (1.0 - 0.3 * dx - 0.2 * dy)

    @classmethod
    def setUpClass(cls):
        cls.cam = make_camera(fx=cls.FX, fy=cls.FY, cx=cls.CX, cy=cls.CY)
        depths = [cls.plane_depth(px, py)
                  for py in range(cls.HEIGHT) for px in range(cls.WIDTH)]
        cls.dmap = DepthMap(cls.WIDTH, cls.HEIGHT, depths)

    def point_on_ray(self, px, py, scale=1.0):
        dx = (px - self.CX) / self.FX
        dy = (py - self.CY) / self.FY
        t = self.plane_depth(px, py) * scale
        return (t * dx, t * dy, t)

    def test_point_on_plane_is_visible(self):
        # Projection lands exactly on pixel (3, 4): bilinear reduces to
        # that pixel's depth, so a near-zero tolerance must suffice.
        verdict = classify_point(self.point_on_ray(3, 4), self.cam, self.dmap,
                                 abs_tol=1e-9)
        self.assertEqual(verdict.status, Status.VISIBLE)
        self.assertAlmostEqual(verdict.z, verdict.depth, places=9)

    def test_point_behind_plane_is_occluded(self):
        verdict = classify_point(self.point_on_ray(3, 4, scale=1.2),
                                 self.cam, self.dmap, abs_tol=1e-9)
        self.assertEqual(verdict.status, Status.OCCLUDED)

    def test_point_in_front_of_plane(self):
        verdict = classify_point(self.point_on_ray(3, 4, scale=0.8),
                                 self.cam, self.dmap, abs_tol=1e-9)
        self.assertEqual(verdict.status, Status.IN_FRONT)


class TestAggregation(unittest.TestCase):
    def test_visible_in_at_least_k(self):
        V, O, F, U = (Status.VISIBLE, Status.OCCLUDED,
                      Status.IN_FRONT, Status.UNKNOWN)
        rows = [
            [V, V, O],   # point 0: visible in 2 views
            [V, O, O],   # point 1: visible in 1 view
            [O, O, O],   # point 2: none
            [V, V, V],   # point 3: visible in 3 views
            [F, U, V],   # point 4: visible in 1 view
        ]
        self.assertEqual(points_visible_in_at_least(rows, 1), {0, 1, 3, 4})
        self.assertEqual(points_visible_in_at_least(rows, 2), {0, 3})
        self.assertEqual(points_visible_in_at_least(rows, 3), {3})
        # k larger than the view count: provably empty, not an error.
        self.assertEqual(points_visible_in_at_least(rows, 4), set())

    def test_aggregation_does_not_rewrite_statuses(self):
        cam = make_camera()
        dmap = const_map(4, 4, 5.0)
        points = [(1.5 * 6.0, 1.5 * 6.0, 6.0)]  # occluded in every view
        result = classify_points(points, [(cam, dmap), (cam, dmap)])
        row = [verdict.status for verdict in result[0]]
        self.assertEqual(row, [Status.OCCLUDED, Status.OCCLUDED])
        self.assertEqual(points_visible_in_at_least([row], 1), set())

    def test_classify_points_matrix_shape_and_independence(self):
        cam_a = make_camera()
        cam_b = make_camera(translation=(0.0, 0.0, -2.0))  # sees z-2
        dmap = const_map(4, 4, 5.0)
        points = [(1.5 * 5.0, 1.5 * 5.0, 5.0),   # z=5: visible A, in-front B
                  (1.5 * 7.0, 1.5 * 7.0, 7.0)]   # z=7: occluded A, visible B
        result = classify_points(points, [(cam_a, dmap), (cam_b, dmap)])
        self.assertEqual(len(result), 2)
        self.assertEqual([len(row) for row in result], [2, 2])
        self.assertEqual(result[0][0].status, Status.VISIBLE)
        self.assertEqual(result[0][1].status, Status.IN_FRONT)
        self.assertEqual(result[1][0].status, Status.OCCLUDED)
        self.assertEqual(result[1][1].status, Status.VISIBLE)
        rows = [[verdict.status for verdict in row] for row in result]
        self.assertEqual(points_visible_in_at_least(rows, 2), set())
        self.assertEqual(points_visible_in_at_least(rows, 1), {0, 1})

    def test_batch_is_atomic_and_pure(self):
        cam = make_camera()
        dmap = const_map(4, 4, 5.0)
        good = (1.5 * 5.0, 1.5 * 5.0, 5.0)
        points = [good, (float("nan"), 0.0, 1.0)]
        views = [(cam, dmap)]
        with self.assertRaises(VisibilityError):
            classify_points(points, views)
        # Inputs untouched by the failed call.
        self.assertEqual(points[0], good)
        self.assertEqual(len(views), 1)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.cam = make_camera()
        self.dmap = const_map(4, 4, 5.0)

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(VisibilityError):
                classify_point((bad, 0.0, 1.0), self.cam, self.dmap)
            with self.assertRaises(VisibilityError):
                DepthMap(2, 2, [1.0, bad, 1.0, 1.0])
            with self.assertRaises(VisibilityError):
                make_camera(fx=bad)
            with self.assertRaises(VisibilityError):
                classify_point((0.0, 0.0, 5.0), self.cam, self.dmap,
                               abs_tol=bad)

    def test_bool_rejected_for_int_params(self):
        with self.assertRaises(VisibilityError):
            DepthMap(True, 4, [1.0] * 4)
        with self.assertRaises(VisibilityError):
            DepthMap(4, False, [1.0] * 4)
        with self.assertRaises(VisibilityError):
            points_visible_in_at_least([[Status.VISIBLE]], True)

    def test_depth_map_contract(self):
        with self.assertRaises(VisibilityError):
            DepthMap(0, 4, [])                       # width below 1
        with self.assertRaises(VisibilityError):
            DepthMap(33, 1, [1.0] * 33)              # width above 32
        with self.assertRaises(VisibilityError):
            DepthMap(2, 2, [1.0, 0.0, 1.0, 1.0])     # zero is no placeholder
        with self.assertRaises(VisibilityError):
            DepthMap(2, 2, [1.0, -1.0, 1.0, 1.0])    # negative depth
        with self.assertRaises(VisibilityError):
            DepthMap(2, 2, [1.0, 1.0])               # wrong length
        with self.assertRaises(VisibilityError):
            DepthMap(2, 2, [1.0] * 4, [False] * 3)   # mask length
        with self.assertRaises(VisibilityError):
            DepthMap(2, 2, [1.0] * 4, [False, True, 1, False])  # non-bool

    def test_camera_contract(self):
        with self.assertRaises(VisibilityError):
            make_camera(fx=0.0)
        with self.assertRaises(VisibilityError):
            make_camera(fy=-1.0)
        with self.assertRaises(VisibilityError):
            make_camera(rotation=((1.0, 0.0), (0.0, 1.0)))       # not 3x3
        with self.assertRaises(VisibilityError):
            make_camera(translation=(0.0, 0.0))                  # not 3
        with self.assertRaises(VisibilityError):
            make_camera(rotation=((1.0, 0.0, float("nan")),
                                  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    def test_tolerance_and_type_contract(self):
        with self.assertRaises(VisibilityError):
            classify_point((0.0, 0.0, 5.0), self.cam, self.dmap,
                           rel_tol=-0.1)
        with self.assertRaises(VisibilityError):
            classify_point((0.0, 0.0), self.cam, self.dmap)      # not 3D
        with self.assertRaises(VisibilityError):
            classify_point((0.0, 0.0, 5.0), "not a camera", self.dmap)
        with self.assertRaises(VisibilityError):
            classify_point((0.0, 0.0, 5.0), self.cam, "not a map")
        with self.assertRaises(VisibilityError):
            points_visible_in_at_least([["visible"]], 1)         # not Status
        with self.assertRaises(VisibilityError):
            points_visible_in_at_least([[Status.VISIBLE]], 0)    # k < 1
        with self.assertRaises(VisibilityError):
            classify_points([(0.0, 0.0, 5.0)], [(self.cam,)])    # bad pair

    def test_error_message_locates_argument(self):
        try:
            DepthMap(2, 2, [1.0, float("nan"), 1.0, 1.0])
        except VisibilityError as exc:
            self.assertIn("depths[1]", str(exc))
        else:
            self.fail("expected VisibilityError")


if __name__ == "__main__":
    unittest.main()
