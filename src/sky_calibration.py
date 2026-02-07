"""Sky calibration system for PTZ camera compass alignment.

Points the camera at the night sky, detects stars, matches them against a
known star catalog via plate solving, and computes the offset between the
camera's reported azimuth and true astronomical north.

Requires: numpy, opencv-python (cv2), requests (for catalog download)
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CatalogStar:
    """A star from the HYG catalog."""
    id: int
    name: str
    ra_rad: float      # Right ascension in radians
    dec_rad: float      # Declination in radians
    magnitude: float


@dataclass
class DetectedStar:
    """A star detected in a camera image."""
    x: float           # Pixel x coordinate
    y: float           # Pixel y coordinate
    brightness: float  # Integrated brightness (sum of pixel values)
    radius: float      # Approximate radius in pixels


@dataclass
class StarMatch:
    """A matched pair of detected and catalog stars."""
    detected: DetectedStar
    catalog: CatalogStar
    alt: float          # Altitude of catalog star (radians)
    az: float           # Azimuth of catalog star (radians)
    residual: float = 0.0  # Pixel residual after fit


@dataclass
class CalibrationResult:
    """Result of a sky calibration run."""
    azimuth_offset: float       # Degrees to add to camera azimuth for true north
    elevation_offset: float     # Degrees to add to camera elevation
    confidence: float           # 0-1, based on number of matches and residuals
    num_matched: int            # Number of star matches used
    rms_residual: float         # RMS pixel residual of the fit
    camera_azimuth: float       # Camera-reported azimuth during calibration
    camera_elevation: float     # Camera-reported elevation during calibration
    timestamp: str              # ISO timestamp of calibration
    matches: list[StarMatch] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Azimuth offset:   {self.azimuth_offset:+.2f}°\n"
            f"Elevation offset: {self.elevation_offset:+.2f}°\n"
            f"Confidence:       {self.confidence:.1%}\n"
            f"Stars matched:    {self.num_matched}\n"
            f"RMS residual:     {self.rms_residual:.1f} px\n"
            f"Camera reported:  az={self.camera_azimuth:.1f}° el={self.camera_elevation:.1f}°\n"
            f"True pointing:    az={self.camera_azimuth + self.azimuth_offset:.1f}° "
            f"el={self.camera_elevation + self.elevation_offset:.1f}°"
        )


# ---------------------------------------------------------------------------
# AstronomyEngine — coordinate transforms
# ---------------------------------------------------------------------------

class AstronomyEngine:
    """Pure-numpy astronomical coordinate transforms.

    All methods are static. Angles in radians unless noted.
    Reference: Meeus, "Astronomical Algorithms", 2nd ed.
    """

    @staticmethod
    def julian_date(dt: datetime) -> float:
        """Compute Julian Date from a UTC datetime."""
        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        y = dt.year
        m = dt.month
        d = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24.0

        if m <= 2:
            y -= 1
            m += 12

        A = int(y / 100)
        B = 2 - A + int(A / 4)

        return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5

    @staticmethod
    def gmst(jd: float) -> float:
        """Greenwich Mean Sidereal Time in radians.

        Args:
            jd: Julian date

        Returns:
            GMST in radians (0 to 2*pi)
        """
        # Centuries since J2000.0
        T = (jd - 2451545.0) / 36525.0
        # GMST in seconds of time (Meeus eq. 12.4)
        theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
                 + 0.000387933 * T * T - T * T * T / 38710000.0)
        return np.deg2rad(theta % 360.0)

    @staticmethod
    def lst(jd: float, lon_rad: float) -> float:
        """Local Sidereal Time in radians.

        Args:
            jd: Julian date
            lon_rad: Observer longitude in radians (east positive)

        Returns:
            LST in radians (0 to 2*pi)
        """
        return (AstronomyEngine.gmst(jd) + lon_rad) % (2 * np.pi)

    @staticmethod
    def radec_to_altaz(ra: np.ndarray, dec: np.ndarray,
                       jd: float, lat_rad: float, lon_rad: float):
        """Convert RA/Dec to Altitude/Azimuth.

        Vectorized — works on arrays of stars.

        Args:
            ra: Right ascension in radians (array or scalar)
            dec: Declination in radians (array or scalar)
            jd: Julian date
            lat_rad: Observer latitude in radians
            lon_rad: Observer longitude in radians

        Returns:
            (alt, az) in radians. Azimuth measured from north through east.
        """
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)

        local_st = AstronomyEngine.lst(jd, lon_rad)
        ha = local_st - ra  # Hour angle

        sin_alt = (np.sin(dec) * np.sin(lat_rad)
                   + np.cos(dec) * np.cos(lat_rad) * np.cos(ha))
        alt = np.arcsin(np.clip(sin_alt, -1, 1))

        cos_az_num = (np.sin(dec) - np.sin(alt) * np.sin(lat_rad))
        cos_az_den = np.cos(alt) * np.cos(lat_rad)
        # Avoid division by zero at poles / zenith
        cos_az = np.where(np.abs(cos_az_den) > 1e-10,
                          cos_az_num / cos_az_den,
                          0.0)
        cos_az = np.clip(cos_az, -1, 1)
        az = np.arccos(cos_az)

        # Azimuth quadrant: if sin(ha) > 0, az = 2*pi - az
        az = np.where(np.sin(ha) > 0, 2 * np.pi - az, az)

        return alt, az

    @staticmethod
    def precess_j2000_to_now(ra_j2000: np.ndarray, dec_j2000: np.ndarray,
                              jd: float):
        """Precess coordinates from J2000.0 to the epoch of jd.

        Uses the IAU 2006 precession (simplified Lieske expressions).

        Args:
            ra_j2000: J2000 right ascension in radians
            dec_j2000: J2000 declination in radians
            jd: Target epoch Julian date

        Returns:
            (ra_now, dec_now) in radians
        """
        ra_j2000 = np.asarray(ra_j2000, dtype=np.float64)
        dec_j2000 = np.asarray(dec_j2000, dtype=np.float64)

        T = (jd - 2451545.0) / 36525.0

        # Precession angles in arcseconds (Lieske 1979)
        zeta_A = (0.6406161 * T + 0.0000839 * T * T
                  + 0.0000050 * T * T * T)
        z_A = (0.6406161 * T + 0.0003041 * T * T
               + 0.0000051 * T * T * T)
        theta_A = (0.5567530 * T - 0.0001185 * T * T
                   - 0.0000116 * T * T * T)

        # Convert to radians (values are in degrees from the polynomial)
        zeta = np.deg2rad(zeta_A)
        z = np.deg2rad(z_A)
        theta = np.deg2rad(theta_A)

        # Rotation
        cos_dec = np.cos(dec_j2000)
        sin_dec = np.sin(dec_j2000)
        cos_ra_zeta = np.cos(ra_j2000 + zeta)
        sin_ra_zeta = np.sin(ra_j2000 + zeta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        A = cos_dec * sin_ra_zeta
        B = cos_theta * cos_dec * cos_ra_zeta - sin_theta * sin_dec
        C = sin_theta * cos_dec * cos_ra_zeta + cos_theta * sin_dec

        ra_now = np.arctan2(A, B) + z
        dec_now = np.arcsin(np.clip(C, -1, 1))

        ra_now = ra_now % (2 * np.pi)
        return ra_now, dec_now

    @staticmethod
    def atmospheric_refraction(alt_rad: float) -> float:
        """Approximate atmospheric refraction correction.

        Args:
            alt_rad: True altitude in radians

        Returns:
            Refraction correction in radians (add to true altitude
            to get apparent altitude)
        """
        alt_deg = np.degrees(alt_rad)
        if alt_deg < -0.5:
            return 0.0
        # Bennett formula (Meeus eq. 16.4), result in arcminutes
        if alt_deg < 0.1:
            alt_deg = 0.1  # Avoid singularity near horizon
        R = 1.0 / np.tan(np.deg2rad(alt_deg + 7.31 / (alt_deg + 4.4)))
        # R is in arcminutes, convert to radians
        return np.deg2rad(R / 60.0)


# ---------------------------------------------------------------------------
# StarCatalog — load and query the HYG database
# ---------------------------------------------------------------------------

class StarCatalog:
    """Star catalog based on the HYG Database v3.

    Auto-downloads the catalog CSV on first use.
    Filters to naked-eye visible stars (mag <= 6.0).
    """

    HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/refs/heads/master/hyg/v3/hyg_v3.csv"
    DEFAULT_MAG_LIMIT = 6.0

    def __init__(self, catalog_path: Optional[str] = None,
                 mag_limit: float = DEFAULT_MAG_LIMIT):
        """Initialize the star catalog.

        Args:
            catalog_path: Path to HYG CSV file. Defaults to data/hygdata_v3.csv
            mag_limit: Maximum magnitude to include (default 6.0)
        """
        if catalog_path is None:
            project_root = Path(__file__).parent.parent
            catalog_path = str(project_root / "data" / "hygdata_v3.csv")
        self._path = catalog_path
        self._mag_limit = mag_limit
        self._stars: list[CatalogStar] = []
        # Arrays for vectorized operations
        self._ra_rad: Optional[np.ndarray] = None
        self._dec_rad: Optional[np.ndarray] = None
        self._mag: Optional[np.ndarray] = None

    @property
    def stars(self) -> list[CatalogStar]:
        return self._stars

    def load(self) -> int:
        """Load the star catalog, downloading if necessary.

        Returns:
            Number of stars loaded
        """
        path = Path(self._path)
        if not path.exists():
            self._download(path)

        self._stars = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mag = float(row.get('mag', '99'))
                    if mag > self._mag_limit:
                        continue

                    # Use rarad/decrad columns if available, else convert
                    ra_rad = row.get('rarad', '')
                    dec_rad = row.get('decrad', '')
                    if ra_rad and dec_rad:
                        ra_r = float(ra_rad)
                        dec_r = float(dec_rad)
                    else:
                        ra_hours = float(row.get('ra', 0))
                        dec_deg = float(row.get('dec', 0))
                        ra_r = ra_hours * (np.pi / 12.0)
                        dec_r = np.deg2rad(dec_deg)

                    star_id = int(row.get('id', 0))
                    name = row.get('proper', '') or row.get('bf', '') or f"HYG-{star_id}"

                    self._stars.append(CatalogStar(
                        id=star_id,
                        name=name,
                        ra_rad=ra_r,
                        dec_rad=dec_r,
                        magnitude=mag,
                    ))
                except (ValueError, KeyError):
                    continue

        # Sort by magnitude (brightest first)
        self._stars.sort(key=lambda s: s.magnitude)

        # Build numpy arrays for vectorized operations
        self._ra_rad = np.array([s.ra_rad for s in self._stars])
        self._dec_rad = np.array([s.dec_rad for s in self._stars])
        self._mag = np.array([s.magnitude for s in self._stars])

        print(f"Loaded {len(self._stars)} stars (mag <= {self._mag_limit})")
        return len(self._stars)

    def _download(self, path: Path):
        """Download the HYG catalog."""
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HYG star catalog to {path}...")
        print(f"  URL: {self.HYG_URL}")

        response = requests.get(self.HYG_URL, timeout=60, stream=True)
        response.raise_for_status()

        total = 0
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)

        print(f"  Downloaded {total / 1024 / 1024:.1f} MB")

    def get_visible_stars(self, jd: float, lat_rad: float, lon_rad: float,
                          center_alt: float, center_az: float,
                          fov_h: float, fov_v: float,
                          max_stars: int = 200,
                          min_alt: float = 0.17) -> list[tuple[CatalogStar, float, float]]:
        """Get catalog stars visible in a camera field of view.

        Args:
            jd: Julian date
            lat_rad: Observer latitude (radians)
            lon_rad: Observer longitude (radians)
            center_alt: Camera boresight altitude (radians)
            center_az: Camera boresight azimuth (radians)
            fov_h: Horizontal field of view (radians)
            fov_v: Vertical field of view (radians)
            max_stars: Maximum stars to return
            min_alt: Minimum altitude above horizon (radians, default ~10°)

        Returns:
            List of (CatalogStar, alt, az) tuples, sorted by magnitude
        """
        if self._ra_rad is None:
            raise RuntimeError("Catalog not loaded. Call load() first.")

        # Precess to current epoch
        ra_now, dec_now = AstronomyEngine.precess_j2000_to_now(
            self._ra_rad, self._dec_rad, jd)

        # Convert all stars to alt/az
        alt_all, az_all = AstronomyEngine.radec_to_altaz(
            ra_now, dec_now, jd, lat_rad, lon_rad)

        # Filter: above minimum altitude
        above_horizon = alt_all > min_alt

        # Filter: within FoV (with 20% margin for edge effects)
        margin = 1.2
        half_h = fov_h * margin / 2
        half_v = fov_v * margin / 2

        # Angular distance from boresight in alt and az
        dalt = alt_all - center_alt
        # Azimuth wraps, so use smallest angular difference
        daz = az_all - center_az
        daz = np.arctan2(np.sin(daz), np.cos(daz))  # Wrap to [-pi, pi]
        # Scale by cos(alt) to account for convergence at zenith
        daz_scaled = daz * np.cos(center_alt)

        in_fov = (np.abs(dalt) < half_v) & (np.abs(daz_scaled) < half_h)

        mask = above_horizon & in_fov
        indices = np.where(mask)[0]

        # Sort by magnitude (brightest first)
        mag_sorted = indices[np.argsort(self._mag[indices])]

        result = []
        for idx in mag_sorted[:max_stars]:
            result.append((self._stars[idx], float(alt_all[idx]), float(az_all[idx])))

        return result


# ---------------------------------------------------------------------------
# CameraModel — PTZ425DB-AT optics
# ---------------------------------------------------------------------------

class CameraModel:
    """Optical model for the Empire Tech PTZ425DB-AT camera.

    1/2.8" STARVIS sensor, 5-125mm zoom (25x).
    """

    # Sensor dimensions for 1/2.8" format (mm)
    SENSOR_W = 5.14
    SENSOR_H = 2.89

    # Focal length range (mm)
    FOCAL_MIN = 5.0    # 1x zoom
    FOCAL_MAX = 125.0  # 25x zoom

    # Default image resolution (main stream)
    IMAGE_W = 2560
    IMAGE_H = 1440

    def __init__(self, zoom: float = 1.0,
                 image_w: int = IMAGE_W, image_h: int = IMAGE_H):
        """Initialize camera model.

        Args:
            zoom: Zoom level (1.0 = wide, 25.0 = full tele)
            image_w: Image width in pixels
            image_h: Image height in pixels
        """
        self.zoom = max(1.0, min(25.0, zoom))
        self.image_w = image_w
        self.image_h = image_h

    @property
    def focal_length(self) -> float:
        """Effective focal length in mm."""
        return self.FOCAL_MIN * self.zoom

    @property
    def fov_h(self) -> float:
        """Horizontal field of view in radians."""
        return 2 * math.atan(self.SENSOR_W / (2 * self.focal_length))

    @property
    def fov_v(self) -> float:
        """Vertical field of view in radians."""
        return 2 * math.atan(self.SENSOR_H / (2 * self.focal_length))

    def sky_to_pixel(self, alt: np.ndarray, az: np.ndarray,
                     center_alt: float, center_az: float):
        """Project sky coordinates to pixel coordinates using gnomonic projection.

        Args:
            alt: Star altitudes in radians (array or scalar)
            az: Star azimuths in radians (array or scalar)
            center_alt: Camera boresight altitude (radians)
            center_az: Camera boresight azimuth (radians)

        Returns:
            (x, y) pixel coordinates. NaN for stars behind camera.
        """
        alt = np.asarray(alt, dtype=np.float64)
        az = np.asarray(az, dtype=np.float64)

        # Convert alt/az to unit vectors on the celestial sphere
        # x = south, y = east, z = up (standard horizontal coords)
        # Then rotate so camera boresight is along the optical axis

        # Direction cosines of star
        cos_alt = np.cos(alt)
        sin_alt = np.sin(alt)
        cos_az = np.cos(az)
        sin_az = np.sin(az)

        # Star in horizontal cartesian (x=N, y=E, z=Up)
        sx = cos_alt * cos_az
        sy = cos_alt * sin_az
        sz = sin_alt

        # Camera boresight direction
        cos_ca = np.cos(center_alt)
        sin_ca = np.sin(center_alt)
        cos_cz = np.cos(center_az)
        sin_cz = np.sin(center_az)

        # Boresight unit vector
        bx = cos_ca * cos_cz
        by = cos_ca * sin_cz
        bz = sin_ca

        # Camera coordinate system:
        # optical axis = boresight
        # "right" in image = east component at boresight
        # "up" in image = perpendicular to both

        # Right vector (tangent to azimuth direction)
        rx = -sin_cz
        ry = cos_cz
        rz = 0.0

        # Up vector (perpendicular to boresight and right)
        ux = -sin_ca * cos_cz
        uy = -sin_ca * sin_cz
        uz = cos_ca

        # Project star onto camera frame
        # dot product with boresight (depth)
        d = sx * bx + sy * by + sz * bz

        # Gnomonic projection: tangent plane
        # Stars behind camera have d <= 0
        valid = d > 0.01

        # Tangent plane coordinates
        xi = np.where(valid, (sx * rx + sy * ry + sz * rz) / d, np.nan)
        eta = np.where(valid, (sx * ux + sy * uy + sz * uz) / d, np.nan)

        # Convert to pixels
        # Pixel scale: radians per pixel
        scale_x = self.fov_h / self.image_w
        scale_y = self.fov_v / self.image_h

        px = self.image_w / 2 + xi / scale_x
        py = self.image_h / 2 - eta / scale_y  # Flip y (image y is down)

        return px, py

    def pixel_to_sky(self, px: np.ndarray, py: np.ndarray,
                     center_alt: float, center_az: float):
        """Convert pixel coordinates back to sky coordinates.

        Args:
            px: Pixel x coordinates
            py: Pixel y coordinates
            center_alt: Camera boresight altitude (radians)
            center_az: Camera boresight azimuth (radians)

        Returns:
            (alt, az) in radians
        """
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)

        scale_x = self.fov_h / self.image_w
        scale_y = self.fov_v / self.image_h

        xi = (px - self.image_w / 2) * scale_x
        eta = -(py - self.image_h / 2) * scale_y

        # Camera frame unit vectors
        cos_ca = np.cos(center_alt)
        sin_ca = np.sin(center_alt)
        cos_cz = np.cos(center_az)
        sin_cz = np.sin(center_az)

        bx = cos_ca * cos_cz
        by = cos_ca * sin_cz
        bz = sin_ca

        rx = -sin_cz
        ry = cos_cz
        rz = 0.0

        ux = -sin_ca * cos_cz
        uy = -sin_ca * sin_cz
        uz = cos_ca

        # Reconstruct direction vector
        dx = bx + xi * rx + eta * ux
        dy = by + xi * ry + eta * uy
        dz = bz + xi * rz + eta * uz

        # Normalize
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= norm
        dy /= norm
        dz /= norm

        alt = np.arcsin(np.clip(dz, -1, 1))
        az = np.arctan2(dy, dx) % (2 * np.pi)

        return alt, az


# ---------------------------------------------------------------------------
# StarDetector — OpenCV star detection
# ---------------------------------------------------------------------------

class StarDetector:
    """Detect stars in night sky images using OpenCV.

    Pipeline: grayscale → blur → background subtract → threshold →
    contour detection → sub-pixel centroids → sort by brightness.
    """

    def __init__(self, blur_size: int = 3, threshold_sigma: float = 3.0,
                 min_area: int = 3, max_area: int = 500,
                 max_stars: int = 100):
        """Initialize detector.

        Args:
            blur_size: Gaussian blur kernel size (odd number)
            threshold_sigma: Threshold = mean + sigma * std
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area in pixels
            max_stars: Maximum number of stars to return
        """
        self.blur_size = blur_size
        self.threshold_sigma = threshold_sigma
        self.min_area = min_area
        self.max_area = max_area
        self.max_stars = max_stars

    def detect(self, image: np.ndarray) -> list[DetectedStar]:
        """Detect stars in an image.

        Args:
            image: BGR or grayscale image (numpy array)

        Returns:
            List of DetectedStar, sorted by brightness (brightest first)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gentle blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # Background estimation using large median filter
        # This handles light pollution gradients
        bg_size = 51
        background = cv2.medianBlur(blurred, bg_size)

        # Subtract background
        subtracted = cv2.subtract(blurred, background)

        # Adaptive threshold based on statistics of the subtracted image
        mean_val = np.mean(subtracted)
        std_val = np.std(subtracted)
        thresh_val = mean_val + self.threshold_sigma * std_val
        thresh_val = max(thresh_val, 15)  # Minimum threshold

        _, binary = cv2.threshold(subtracted, int(thresh_val), 255,
                                  cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        stars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            # Compute moments for centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Sub-pixel refinement using weighted centroid on original image
            x_min = max(0, int(cx) - 5)
            x_max = min(gray.shape[1], int(cx) + 6)
            y_min = max(0, int(cy) - 5)
            y_max = min(gray.shape[0], int(cy) + 6)

            roi = gray[y_min:y_max, x_min:x_max].astype(np.float64)
            bg_local = float(np.median(roi))
            roi_sub = np.maximum(roi - bg_local, 0)

            total = np.sum(roi_sub)
            if total < 1:
                continue

            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            sub_cx = np.sum(xx * roi_sub) / total
            sub_cy = np.sum(yy * roi_sub) / total

            radius = math.sqrt(area / math.pi)

            stars.append(DetectedStar(
                x=sub_cx,
                y=sub_cy,
                brightness=total,
                radius=radius,
            ))

        # Sort by brightness (brightest first)
        stars.sort(key=lambda s: s.brightness, reverse=True)
        return stars[:self.max_stars]


# ---------------------------------------------------------------------------
# PlateSolver — triangle hash matching
# ---------------------------------------------------------------------------

class PlateSolver:
    """Plate solver using triangle hash matching (Groth 1986).

    Forms triangles from the brightest N stars in both the detected and
    catalog sets, computes scale-invariant hashes (side ratios), matches
    by proximity in hash space, votes on correspondences, and computes
    the best-fit affine transform.
    """

    def __init__(self, n_stars: int = 15, hash_tolerance: float = 0.03,
                 min_matches: int = 4):
        """Initialize plate solver.

        Args:
            n_stars: Number of brightest stars to use for triangle matching
            hash_tolerance: Tolerance for hash matching (fraction of side ratio)
            min_matches: Minimum number of star matches required
        """
        self.n_stars = n_stars
        self.hash_tolerance = hash_tolerance
        self.min_matches = min_matches

    @staticmethod
    def _triangle_hash(x1: float, y1: float, x2: float, y2: float,
                       x3: float, y3: float):
        """Compute scale-invariant triangle hash.

        Returns:
            (ratio1, ratio2, vertex_order) where ratio1 = short/long,
            ratio2 = medium/long side ratios, and vertex_order is the
            indices sorted by opposing side length.
        """
        d12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        d23 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        d13 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        sides = [(d23, 0), (d13, 1), (d12, 2)]  # (length, opposing vertex)
        sides.sort(key=lambda s: s[0])

        longest = sides[2][0]
        if longest < 1e-6:
            return None

        ratio1 = sides[0][0] / longest  # short/long
        ratio2 = sides[1][0] / longest  # medium/long

        vertex_order = (sides[0][1], sides[1][1], sides[2][1])

        return ratio1, ratio2, vertex_order

    def _build_triangle_hashes(self, points: list[tuple[float, float]]):
        """Build triangle hash table from a set of points.

        Args:
            points: List of (x, y) coordinates

        Returns:
            List of (ratio1, ratio2, (i, j, k), vertex_order) tuples
        """
        n = min(len(points), self.n_stars)
        hashes = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    result = self._triangle_hash(
                        points[i][0], points[i][1],
                        points[j][0], points[j][1],
                        points[k][0], points[k][1],
                    )
                    if result is not None:
                        ratio1, ratio2, vertex_order = result
                        hashes.append((ratio1, ratio2, (i, j, k), vertex_order))
        return hashes

    def solve(self, detected: list[DetectedStar],
              catalog_projected: list[tuple[CatalogStar, float, float]]):
        """Match detected stars to catalog stars projected onto the image.

        Args:
            detected: Detected stars (pixel coordinates)
            catalog_projected: List of (CatalogStar, pixel_x, pixel_y) tuples

        Returns:
            List of (detected_index, catalog_index) matched pairs,
            or empty list if solving fails.
        """
        if len(detected) < 3 or len(catalog_projected) < 3:
            return []

        # Extract points
        det_points = [(s.x, s.y) for s in detected[:self.n_stars]]
        cat_points = [(px, py) for _, px, py in catalog_projected[:self.n_stars]]

        # Build triangle hashes
        det_hashes = self._build_triangle_hashes(det_points)
        cat_hashes = self._build_triangle_hashes(cat_points)

        if not det_hashes or not cat_hashes:
            return []

        # Match triangles by hash proximity
        # Vote on star correspondences
        n_det = len(det_points)
        n_cat = len(cat_points)
        votes = np.zeros((n_det, n_cat), dtype=np.int32)

        for dr1, dr2, d_idx, d_vo in det_hashes:
            for cr1, cr2, c_idx, c_vo in cat_hashes:
                if (abs(dr1 - cr1) < self.hash_tolerance
                        and abs(dr2 - cr2) < self.hash_tolerance):
                    # Triangle match — vote for vertex correspondences
                    # Map detected vertices to catalog vertices using
                    # vertex_order (sorted by opposing side length)
                    for vi in range(3):
                        d_star = d_idx[d_vo[vi]]
                        c_star = c_idx[c_vo[vi]]
                        votes[d_star, c_star] += 1

        # Extract correspondences from vote matrix
        matches = []
        used_det = set()
        used_cat = set()

        # Greedy: highest votes first
        while True:
            max_val = votes.max()
            if max_val < 2:  # Require at least 2 triangle votes
                break
            d_idx, c_idx = np.unravel_index(votes.argmax(), votes.shape)
            d_idx, c_idx = int(d_idx), int(c_idx)
            if d_idx not in used_det and c_idx not in used_cat:
                matches.append((d_idx, c_idx))
                used_det.add(d_idx)
                used_cat.add(c_idx)
            votes[d_idx, c_idx] = 0

        if len(matches) < self.min_matches:
            return []

        # Refine: compute affine transform and reject outliers
        matches = self._refine_matches(matches, det_points, cat_points)

        return matches

    def _refine_matches(self, matches: list[tuple[int, int]],
                        det_points: list[tuple[float, float]],
                        cat_points: list[tuple[float, float]],
                        max_residual: float = 30.0) -> list[tuple[int, int]]:
        """Refine matches by fitting an affine transform and rejecting outliers.

        Args:
            matches: Initial (detected_idx, catalog_idx) pairs
            det_points: Detected star pixel positions
            cat_points: Catalog star pixel positions
            max_residual: Maximum allowed residual in pixels

        Returns:
            Refined list of matches
        """
        if len(matches) < 3:
            return matches

        # Build point arrays
        src = np.array([[det_points[d][0], det_points[d][1]] for d, _ in matches],
                       dtype=np.float64)
        dst = np.array([[cat_points[c][0], cat_points[c][1]] for _, c in matches],
                       dtype=np.float64)

        # Fit affine transform (least squares)
        # dst = A * src + b
        # Solve for A, b using all matches
        n = len(matches)
        A_mat = np.zeros((2 * n, 6))
        b_vec = np.zeros(2 * n)
        for i in range(n):
            A_mat[2 * i, 0] = src[i, 0]
            A_mat[2 * i, 1] = src[i, 1]
            A_mat[2 * i, 2] = 1
            b_vec[2 * i] = dst[i, 0]

            A_mat[2 * i + 1, 3] = src[i, 0]
            A_mat[2 * i + 1, 4] = src[i, 1]
            A_mat[2 * i + 1, 5] = 1
            b_vec[2 * i + 1] = dst[i, 1]

        try:
            params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return matches

        # Compute residuals
        refined = []
        for d, c in matches:
            pred_x = params[0] * det_points[d][0] + params[1] * det_points[d][1] + params[2]
            pred_y = params[3] * det_points[d][0] + params[4] * det_points[d][1] + params[5]
            residual = math.sqrt((pred_x - cat_points[c][0]) ** 2
                                 + (pred_y - cat_points[c][1]) ** 2)
            if residual < max_residual:
                refined.append((d, c))

        return refined


# ---------------------------------------------------------------------------
# CameraCalibrator — orchestrator
# ---------------------------------------------------------------------------

class CameraCalibrator:
    """Orchestrates sky calibration for a PTZ camera.

    Workflow:
    1. Point camera at a sky position (high elevation, dark region)
    2. Capture a frame
    3. Detect stars in the image
    4. Query the star catalog for expected stars in the FoV
    5. Project catalog stars to pixel coordinates
    6. Plate-solve to find matches
    7. Compute azimuth/elevation offset

    Repeat at multiple sky positions for verification.
    """

    # Default sky positions: azimuths at fixed elevation
    DEFAULT_AZIMUTHS = [0, 90, 180, 270]
    DEFAULT_ELEVATION = 60  # degrees

    def __init__(self, camera, catalog: StarCatalog,
                 lat: float, lon: float,
                 zoom: float = 1.0):
        """Initialize calibrator.

        Args:
            camera: EmpireTechPTZ camera instance (connected)
            catalog: Loaded StarCatalog
            lat: Observer latitude in degrees (north positive)
            lon: Observer longitude in degrees (east positive)
            zoom: Camera zoom level to use (default 1x for wide FoV)
        """
        self.camera = camera
        self.catalog = catalog
        self.lat_rad = np.deg2rad(lat)
        self.lon_rad = np.deg2rad(lon)
        self.lat_deg = lat
        self.lon_deg = lon
        self.camera_model = CameraModel(zoom=zoom)
        self.detector = StarDetector()
        self.solver = PlateSolver()
        self._save_images = False
        self._image_dir: Optional[Path] = None

    def enable_image_saving(self, output_dir: str):
        """Enable saving of captured and annotated images.

        Args:
            output_dir: Directory to save images to
        """
        self._save_images = True
        self._image_dir = Path(output_dir)
        self._image_dir.mkdir(parents=True, exist_ok=True)

    def calibrate_single(self, azimuth: float, elevation: float,
                         timestamp: Optional[datetime] = None
                         ) -> Optional[CalibrationResult]:
        """Run calibration at a single sky position.

        Args:
            azimuth: Camera azimuth to point to (degrees)
            elevation: Camera elevation to point to (degrees)
            timestamp: Override timestamp (default: now)

        Returns:
            CalibrationResult or None if calibration fails
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        print(f"\n--- Calibrating at az={azimuth:.0f}° el={elevation:.0f}° ---")

        # Move camera
        print("  Moving camera...")
        if not self.camera.goto_position(azimuth, elevation, wait=True):
            print("  WARNING: Camera move may not have completed")
            time.sleep(2)

        # Wait a moment for vibration to settle
        time.sleep(1.0)

        # Get actual camera position
        pos = self.camera.get_position()
        if pos is None:
            print("  ERROR: Cannot read camera position")
            return None
        cam_az, cam_el, cam_zoom = pos
        print(f"  Camera reports: az={cam_az:.1f}° el={cam_el:.1f}° zoom={cam_zoom:.1f}x")

        # Capture frame
        print("  Capturing frame...")
        frame = self._capture_frame()
        if frame is None:
            print("  ERROR: Failed to capture frame")
            return None
        print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

        # Update camera model with actual resolution
        self.camera_model.image_w = frame.shape[1]
        self.camera_model.image_h = frame.shape[0]

        # Detect stars
        print("  Detecting stars...")
        detected = self.detector.detect(frame)
        print(f"  Detected {len(detected)} stars")

        if len(detected) < 3:
            print("  ERROR: Too few stars detected (need at least 3)")
            if self._save_images:
                self._save_annotated(frame, detected, [], azimuth, "nosolve")
            return None

        # Compute expected star positions
        jd = AstronomyEngine.julian_date(timestamp)
        center_alt = np.deg2rad(cam_el)
        center_az = np.deg2rad(cam_az)

        visible = self.catalog.get_visible_stars(
            jd, self.lat_rad, self.lon_rad,
            center_alt, center_az,
            self.camera_model.fov_h, self.camera_model.fov_v)
        print(f"  Catalog stars in FoV: {len(visible)}")

        if len(visible) < 3:
            print("  ERROR: Too few catalog stars in FoV")
            return None

        # Project catalog stars to pixel coordinates
        cat_projected = []
        for star, alt, az in visible:
            px, py = self.camera_model.sky_to_pixel(
                np.array([alt]), np.array([az]), center_alt, center_az)
            if not np.isnan(px[0]) and not np.isnan(py[0]):
                cat_projected.append((star, float(px[0]), float(py[0])))

        print(f"  Projected catalog stars: {len(cat_projected)}")

        if len(cat_projected) < 3:
            print("  ERROR: Too few catalog stars project onto image")
            return None

        # Plate solve
        print("  Plate solving...")
        matches_idx = self.solver.solve(detected, cat_projected)
        print(f"  Matched {len(matches_idx)} stars")

        if not matches_idx:
            print("  ERROR: Plate solving failed")
            if self._save_images:
                self._save_annotated(frame, detected, [], azimuth, "nosolve")
            return None

        # Build match list and compute offset
        star_matches = []
        for d_idx, c_idx in matches_idx:
            det_star = detected[d_idx]
            cat_star, cat_px, cat_py = cat_projected[c_idx]
            # Find alt/az from visible list
            cat_alt, cat_az = None, None
            for s, a, z in visible:
                if s.id == cat_star.id:
                    cat_alt, cat_az = a, z
                    break

            residual = math.sqrt((det_star.x - cat_px) ** 2
                                 + (det_star.y - cat_py) ** 2)
            star_matches.append(StarMatch(
                detected=det_star,
                catalog=cat_star,
                alt=cat_alt if cat_alt is not None else 0,
                az=cat_az if cat_az is not None else 0,
                residual=residual,
            ))

        # Compute azimuth and elevation offsets
        # For each matched star, compute what the camera center must truly
        # be pointing at, given the star's known sky position and its pixel
        # position in the image
        az_offsets = []
        el_offsets = []
        for match in star_matches:
            # Where the detected star is in pixels
            px = match.detected.x
            py = match.detected.y
            # Convert pixel back to sky using camera model
            det_alt, det_az = self.camera_model.pixel_to_sky(
                np.array([px]), np.array([py]), center_alt, center_az)
            det_alt = float(det_alt[0])
            det_az = float(det_az[0])

            # The catalog says this star is at (match.alt, match.az)
            # The camera model (using camera-reported boresight) puts it at
            # (det_alt, det_az). The difference tells us the offset.
            daz = match.az - det_az
            daz = math.atan2(math.sin(daz), math.cos(daz))  # Wrap
            dal = match.alt - det_alt

            az_offsets.append(math.degrees(daz))
            el_offsets.append(math.degrees(dal))

        az_offset = float(np.median(az_offsets))
        el_offset = float(np.median(el_offsets))
        rms = math.sqrt(np.mean([m.residual ** 2 for m in star_matches]))

        # Confidence based on number of matches and residual quality
        match_score = min(len(star_matches) / 8.0, 1.0)
        residual_score = max(1.0 - rms / 50.0, 0.0)
        confidence = match_score * 0.6 + residual_score * 0.4

        result = CalibrationResult(
            azimuth_offset=az_offset,
            elevation_offset=el_offset,
            confidence=confidence,
            num_matched=len(star_matches),
            rms_residual=rms,
            camera_azimuth=cam_az,
            camera_elevation=cam_el,
            timestamp=timestamp.isoformat(),
            matches=star_matches,
            metadata={
                "latitude": self.lat_deg,
                "longitude": self.lon_deg,
                "zoom": self.camera_model.zoom,
                "detected_stars": len(detected),
                "catalog_stars_in_fov": len(visible),
                "julian_date": jd,
            }
        )

        print(f"  Result: az_offset={az_offset:+.2f}° el_offset={el_offset:+.2f}° "
              f"confidence={confidence:.1%}")

        if self._save_images:
            self._save_annotated(frame, detected, star_matches, azimuth, "solved")

        return result

    def calibrate(self, azimuths: Optional[list[float]] = None,
                  elevation: float = DEFAULT_ELEVATION,
                  ) -> Optional[CalibrationResult]:
        """Run calibration at multiple sky positions and combine results.

        Args:
            azimuths: List of azimuths to test (degrees). Default: [0, 90, 180, 270]
            elevation: Elevation to use (degrees)

        Returns:
            Combined CalibrationResult, or None if all positions fail
        """
        if azimuths is None:
            azimuths = self.DEFAULT_AZIMUTHS

        results = []
        for az in azimuths:
            result = self.calibrate_single(az, elevation)
            if result is not None:
                results.append(result)

        if not results:
            print("\nERROR: All calibration positions failed")
            return None

        # Combine results: weighted median by confidence
        az_offsets = [r.azimuth_offset for r in results]
        el_offsets = [r.elevation_offset for r in results]
        confidences = [r.confidence for r in results]

        # Weighted average
        weights = np.array(confidences)
        weights /= weights.sum()

        combined_az = float(np.average(az_offsets, weights=weights))
        combined_el = float(np.average(el_offsets, weights=weights))
        combined_conf = float(np.mean(confidences))
        total_matched = sum(r.num_matched for r in results)
        avg_rms = float(np.mean([r.rms_residual for r in results]))

        # Check consistency across positions
        az_std = float(np.std(az_offsets))
        el_std = float(np.std(el_offsets))
        if az_std > 2.0:
            print(f"  WARNING: Azimuth offsets vary by {az_std:.1f}° across positions")
            combined_conf *= 0.5

        combined = CalibrationResult(
            azimuth_offset=combined_az,
            elevation_offset=combined_el,
            confidence=min(combined_conf, 1.0),
            num_matched=total_matched,
            rms_residual=avg_rms,
            camera_azimuth=results[0].camera_azimuth,
            camera_elevation=results[0].camera_elevation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "num_positions": len(results),
                "az_offset_std": az_std,
                "el_offset_std": el_std,
                "individual_results": [
                    {
                        "azimuth": r.camera_azimuth,
                        "az_offset": r.azimuth_offset,
                        "el_offset": r.elevation_offset,
                        "confidence": r.confidence,
                        "num_matched": r.num_matched,
                    }
                    for r in results
                ],
            }
        )

        print(f"\n=== Combined Calibration Result ({len(results)} positions) ===")
        print(combined.summary())

        return combined

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera.

        Takes multiple frames and uses the last one to ensure we get
        a fresh frame after the camera has settled.
        """
        # Grab several frames to flush buffer
        for _ in range(5):
            frame = self.camera.get_latest_frame()
            if frame is None:
                frame_q = self.camera.get_frame(timeout=2.0)
                if frame_q is not None:
                    frame = frame_q
            time.sleep(0.1)

        if frame is None:
            return None

        # Return the image array from Frame dataclass
        if hasattr(frame, 'image'):
            return frame.image
        return frame

    def _save_annotated(self, image: np.ndarray,
                        detected: list[DetectedStar],
                        matches: list[StarMatch],
                        azimuth: float, label: str):
        """Save annotated image for debugging."""
        if self._image_dir is None:
            return

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Draw all detected stars as green circles
        for star in detected:
            cv2.circle(annotated, (int(star.x), int(star.y)),
                       max(int(star.radius * 2), 3), (0, 255, 0), 1)

        # Draw matched stars as red circles with labels
        for match in matches:
            x, y = int(match.detected.x), int(match.detected.y)
            cv2.circle(annotated, (x, y), 8, (0, 0, 255), 2)
            label_text = match.catalog.name[:12]
            cv2.putText(annotated, label_text, (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cal_az{int(azimuth)}_{label}_{ts}.jpg"
        path = self._image_dir / filename
        cv2.imwrite(str(path), annotated)
        print(f"  Saved: {path}")
