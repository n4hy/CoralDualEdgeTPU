#!/usr/bin/env python3
"""Calibrate PTZ camera compass using star field plate solving.

Points the camera at the night sky, detects stars, matches them against the
HYG star catalog via triangle-hash plate solving, and computes the exact
offset between the camera's reported azimuth and true astronomical north.

Usage:
    # Standard calibration (4 sky positions)
    python calibrate_camera.py --lat 39.6477 --lon -76.1347

    # Quick test (1 position)
    python calibrate_camera.py --lat 39.6477 --lon -76.1347 --quick

    # Custom options, save debug images
    python calibrate_camera.py --lat 39.6477 --lon -76.1347 --save-images --positions 5

    # Custom camera IP
    python calibrate_camera.py --lat 39.6477 --lon -76.1347 --camera-ip 192.168.1.200

Before running:
1. Run at night with clear skies
2. Ensure camera is accessible: ping 192.168.1.108
3. Ensure ethernet interface is configured: ip addr show enp97s0
4. Activate the Python environment: source coral39/bin/activate
5. Star catalog will be auto-downloaded on first run (~15MB)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.camera import CameraConfig, EmpireTechPTZ
from src.sky_calibration import CameraCalibrator, StarCatalog

# Default configuration
DEFAULT_CAMERA_IP = "192.168.1.108"
DEFAULT_CAMERA_USER = "admin"
DEFAULT_CAMERA_PASS = "Admin123!"
DEFAULT_LAT = 39.64768815
DEFAULT_LON = -76.13474955
DEFAULT_ELEVATION = 60  # degrees
DEFAULT_ZOOM = 1.0      # 1x = widest FoV
DEFAULT_POSITIONS = 4


def check_camera(ip: str) -> bool:
    """Check if camera is reachable."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", ip],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sky Calibration — PTZ Camera Compass Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--lat", type=float, default=DEFAULT_LAT,
                        help=f"Observer latitude in degrees N (default: {DEFAULT_LAT})")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON,
                        help=f"Observer longitude in degrees E (default: {DEFAULT_LON})")
    parser.add_argument("--camera-ip", type=str, default=DEFAULT_CAMERA_IP,
                        help=f"Camera IP address (default: {DEFAULT_CAMERA_IP})")
    parser.add_argument("--user", "-u", type=str, default=DEFAULT_CAMERA_USER,
                        help="Camera username")
    parser.add_argument("--password", "-p", type=str,
                        default=DEFAULT_CAMERA_PASS, help="Camera password")
    parser.add_argument("--zoom", "-z", type=float, default=DEFAULT_ZOOM,
                        help=f"Zoom level (default: {DEFAULT_ZOOM}, 1x = widest FoV)")
    parser.add_argument("--elevation", "-e", type=float, default=DEFAULT_ELEVATION,
                        help=f"Sky elevation to point at, degrees (default: {DEFAULT_ELEVATION})")
    parser.add_argument("--positions", "-n", type=int, default=DEFAULT_POSITIONS,
                        help=f"Number of sky positions to test (default: {DEFAULT_POSITIONS})")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Path to HYG star catalog CSV (default: data/hygdata_v3.csv)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured and annotated images for debugging")
    parser.add_argument("--quick", action="store_true",
                        help="Quick calibration (1 position only)")
    parser.add_argument("--output-dir", "-o", type=str, default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--skip-camera-check", action="store_true",
                        help="Skip camera connectivity check")

    args = parser.parse_args()

    print("=" * 60)
    print("Sky Calibration — PTZ Camera Compass Alignment")
    print("=" * 60)
    print(f"Observer: {args.lat:.6f}°N, {args.lon:.6f}°E")
    print(f"Camera:   {args.camera_ip}")
    print(f"Zoom:     {args.zoom}x")
    print(f"Elevation: {args.elevation}°")
    print()

    # Check camera connectivity
    if not args.skip_camera_check:
        print(f"Checking camera at {args.camera_ip}...")
        if not check_camera(args.camera_ip):
            print(f"\nERROR: Cannot reach camera at {args.camera_ip}")
            print("\nEnsure:")
            print("  1. Camera is powered on")
            print("  2. Ethernet interface is configured:")
            print("     ip addr show enp97s0")
            print("  3. Camera IP is correct (default: 192.168.1.108)")
            print("\nOr use --skip-camera-check to bypass this check")
            return 1
        print("  Camera reachable")

    # Load star catalog
    print("\nLoading star catalog...")
    catalog = StarCatalog(catalog_path=args.catalog)
    try:
        n_stars = catalog.load()
    except Exception as e:
        print(f"\nERROR: Failed to load star catalog: {e}")
        print("\nThe catalog will be auto-downloaded on first run.")
        print("If download fails, manually download from:")
        print(f"  {StarCatalog.HYG_URL}")
        print("  Save to: data/hygdata_v3.csv")
        return 1

    # Connect camera
    print(f"\nConnecting to camera at {args.camera_ip}...")
    rtsp_url = EmpireTechPTZ.create_rtsp_url(
        args.camera_ip, args.user, args.password, subtype=0)
    config = CameraConfig(
        name="PTZ-Calibration",
        rtsp_url=rtsp_url,
        username=args.user,
        password=args.password,
        resolution=(2560, 1440),
    )
    camera = EmpireTechPTZ(config)

    try:
        if not camera.connect():
            print("\nERROR: Failed to connect to camera RTSP stream")
            print("Check credentials and stream URL")
            return 1

        camera.start()

        # Create calibrator
        calibrator = CameraCalibrator(
            camera=camera,
            catalog=catalog,
            lat=args.lat,
            lon=args.lon,
            zoom=args.zoom,
        )

        if args.save_images:
            output_dir = Path(args.output_dir)
            calibrator.enable_image_saving(str(output_dir / "calibration_images"))

        # Generate sky positions
        if args.quick:
            azimuths = [0.0]
        else:
            step = 360.0 / args.positions
            azimuths = [i * step for i in range(args.positions)]

        print(f"\nCalibrating at {len(azimuths)} position(s): "
              f"{[f'{a:.0f}°' for a in azimuths]}")

        # Run calibration
        result = calibrator.calibrate(
            azimuths=azimuths,
            elevation=args.elevation,
        )

        if result is None:
            print("\nCalibration FAILED — could not solve any positions")
            print("\nTroubleshooting:")
            print("  - Ensure it is nighttime with clear skies")
            print("  - Try a higher elevation (--elevation 70)")
            print("  - Reduce light pollution (turn off nearby lights)")
            print("  - Check camera focus (stars should be point sources)")
            return 1

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON result
        json_data = {
            "calibration_type": "sky_star_field",
            "timestamp": result.timestamp,
            "azimuth_offset_deg": result.azimuth_offset,
            "elevation_offset_deg": result.elevation_offset,
            "confidence": result.confidence,
            "num_matched_stars": result.num_matched,
            "rms_residual_px": result.rms_residual,
            "observer": {
                "latitude": args.lat,
                "longitude": args.lon,
            },
            "camera": {
                "ip": args.camera_ip,
                "zoom": args.zoom,
                "reported_azimuth": result.camera_azimuth,
                "reported_elevation": result.camera_elevation,
            },
            "metadata": result.metadata,
        }

        json_path = output_dir / f"sky_calibration_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Markdown report
        md_path = output_dir / f"sky_calibration_{timestamp}.md"
        md_lines = [
            "# Sky Calibration Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Observer:** {args.lat:.6f}°N, {args.lon:.6f}°E",
            "",
            "## Results",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Azimuth offset | {result.azimuth_offset:+.2f}° |",
            f"| Elevation offset | {result.elevation_offset:+.2f}° |",
            f"| Confidence | {result.confidence:.1%} |",
            f"| Stars matched | {result.num_matched} |",
            f"| RMS residual | {result.rms_residual:.1f} px |",
            "",
            "## Interpretation",
            "",
            f"To convert camera azimuth to true north, "
            f"**add {result.azimuth_offset:+.2f}°** to the camera's reported azimuth.",
            "",
            f"Camera reports az={result.camera_azimuth:.1f}° → "
            f"true az={result.camera_azimuth + result.azimuth_offset:.1f}°",
            "",
        ]

        if "individual_results" in result.metadata:
            md_lines.extend([
                "## Individual Positions",
                "",
                "| Position Az | Az Offset | El Offset | Confidence | Stars |",
                "|------------|-----------|-----------|------------|-------|",
            ])
            for r in result.metadata["individual_results"]:
                md_lines.append(
                    f"| {r['azimuth']:.0f}° | {r['az_offset']:+.2f}° | "
                    f"{r['el_offset']:+.2f}° | {r['confidence']:.1%} | "
                    f"{r['num_matched']} |"
                )
            md_lines.append("")

        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines) + '\n')

        print(f"\n{'='*60}")
        print("CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(result.summary())
        print(f"\nResults saved to:")
        print(f"  JSON:     {json_path}")
        print(f"  Markdown: {md_path}")

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\nCalibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        camera.stop()
        camera.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
