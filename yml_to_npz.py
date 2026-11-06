"""
Konvertiert eine omnicalib-Kalibrierung (``calibration.yml``, Scaramuzza-Modell)
in ein NPZ mit Kameramatrix ``K`` (3x3) und ``cv2.fisheye``-Verzerrung ``D`` (4x1).

Modelle
-------
omnicalib (Feld ``poly_incident_angle_to_radius``, Koeffizienten a0..aN aufsteigend):

    r(theta) = a0 + a1*theta + a2*theta^2 + ... + aN*theta^N
    (u, v)   = principal_point + r(theta) * (x, y) / |(x, y)|

cv2.fisheye (Kannala-Brandt, fx = fy = f, Skew = 0):

    theta_d  = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    r(theta) = f * theta_d
             = c1*theta + c3*theta^3 + c5*theta^5 + c7*theta^7 + c9*theta^9

Da das cv2-Modell nur ungerade Potenzen kennt, werden c1..c9 per linearer
Least-Squares-Anpassung an r(theta) im Bereich [0, theta_max] bestimmt;
danach gilt f = c1 und k_i = c_(2i+1) / c1.

Fit-Bereich theta_max
---------------------
* mit ``--detections``: groesster beobachteter Einfallswinkel der Kalibrierpunkte
  (berechnet aus den Extrinsics der yml), d.h. nur der durch Messdaten gestuetzte Bereich
* sonst: 89 Grad
* ``--max-angle`` ueberschreibt diesen Default.

Zusaetzlich wird theta_max immer begrenzt auf
* < 90 Grad: cv2.fisheye bildet Strahlen ueber ``theta = atan(|(x, y)| / z)`` ab und
  kann daher nur Strahlen mit theta < 90 Grad darstellen,
* den Bereich, in dem r(theta) monoton steigt,
* den Winkel der am weitesten vom Hauptpunkt entfernten Bildecke (falls die
  Bildgroesse bekannt ist).

Pruefung
--------
* immer: cv2.fisheye.projectPoints gegen das omnicalib-Modell auf Zufallsstrahlen
  (Fehler der Modellumrechnung)
* mit ``--detections``: Reprojektionsfehler beider Modelle gegen die detektierten
  Bildpunkte
"""
import argparse
import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml
from numpy.polynomial import Polynomial

THETA_LIMIT = np.radians(89.999)


def load_calibration(path: Path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    poly_theta = np.asarray(data["poly_incident_angle_to_radius"], dtype=np.float64)
    principal_point = np.asarray(data["principal_point"], dtype=np.float64)
    extrinsics = np.asarray(data.get("extrinsics", []), dtype=np.float64)
    return poly_theta, principal_point, extrinsics


def load_detections(path: Path, extrinsics: np.ndarray):
    """
    Laedt ``detections.pickle`` (Format von omnicalib) und ordnet die Bilder wie
    omnicalib (sortierte Bildpfade) den Extrinsics zu.
    Rueckgabe: Liste von (Bildpfad, Weltpunkte Nx3, Bildpunkte Nx2, R 3x3, T 3)
    """
    with open(path, "rb") as f:
        detections = pickle.load(f)["detections"]
    paths = sorted(detections.keys())
    if len(paths) != len(extrinsics):
        raise SystemExit(
            f"Fehler: {len(paths)} Bilder in {path}, aber {len(extrinsics)} Extrinsics "
            "in der yml - passen Detektionen und Kalibrierung zusammen?")
    views = []
    for p, ext in zip(paths, extrinsics):
        wp = np.asarray(detections[p]["object_points"], dtype=np.float64).reshape(-1, 3)
        ip = np.asarray(detections[p]["image_points"], dtype=np.float64).reshape(-1, 2)
        views.append((p, wp, ip, ext[:, :3], ext[:, 3]))
    return views


def image_size_from_file(path) -> tuple:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        return None
    H, W = image.shape
    return W, H


def incident_angle(view_points: np.ndarray) -> np.ndarray:
    """Winkel zwischen den Sichtstrahlen und der optischen Achse (0, 0, 1)."""
    return np.arccos(np.clip(
        view_points[:, 2] / np.linalg.norm(view_points, axis=1), -1.0, 1.0))


def theta_for_radius(poly_theta: np.ndarray, radius: float) -> float:
    """Kleinster positiver Einfallswinkel mit r(theta) = radius (oder inf)."""
    roots = (Polynomial(poly_theta) - radius).roots()
    roots = roots[np.isclose(roots.imag, 0.0)].real
    roots = roots[roots > 0]
    return float(roots.min()) if roots.size else np.inf


def monotonic_limit(poly_theta: np.ndarray) -> float:
    """Erster Winkel > 0, an dem r(theta) nicht mehr monoton steigt (oder inf)."""
    roots = Polynomial(poly_theta).deriv().roots()
    roots = roots[np.isclose(roots.imag, 0.0)].real
    roots = roots[roots > 0]
    return float(roots.min()) if roots.size else np.inf


def fit_fisheye(poly_theta: np.ndarray, theta_max: float, samples: int = 5000):
    """Least-Squares-Fit von f*theta_d(theta) an das omnicalib-Polynom."""
    theta = np.linspace(0.0, theta_max, samples)
    r = Polynomial(poly_theta)(theta)
    A = np.stack([theta ** p for p in (1, 3, 5, 7, 9)], axis=1)
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    f = c[0]
    k = c[1:] / f
    residual = A @ c - r
    return f, k, residual


def project_omnicalib(view_points, poly_theta, principal_point):
    theta = incident_angle(view_points)
    xy = view_points[:, :2]
    norm = np.linalg.norm(xy, axis=1, keepdims=True)
    direction = np.divide(xy, norm, out=np.zeros_like(xy), where=norm > 0)
    return principal_point + Polynomial(poly_theta)(theta)[:, None] * direction


def project_fisheye(view_points, K, D):
    uv, _ = cv.fisheye.projectPoints(
        np.ascontiguousarray(view_points[:, None, :]), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


def verify_synthetic(poly_theta, principal_point, K, D, theta_max, n=20000, seed=0):
    """Vergleicht cv2.fisheye mit dem omnicalib-Modell auf Zufallsstrahlen."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, theta_max, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    rays = np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)
    err = np.linalg.norm(project_fisheye(rays, K, D)
                         - project_omnicalib(rays, poly_theta, principal_point), axis=1)
    return err


def verify_detections(views, poly_theta, principal_point, K, D):
    """Reprojektionsfehler beider Modelle gegen die detektierten Bildpunkte."""
    err_omni, err_cv = [], []
    for _, wp, ip, R, T in views:
        v = wp @ R.T + T
        err_omni.append(np.linalg.norm(
            project_omnicalib(v, poly_theta, principal_point) - ip, axis=1))
        err_cv.append(np.linalg.norm(project_fisheye(v, K, D) - ip, axis=1))
    return np.concatenate(err_omni), np.concatenate(err_cv)


def stats(err: np.ndarray) -> str:
    return (f"RMS {np.sqrt(np.mean(err ** 2)):.4f} px, "
            f"mean {err.mean():.4f} px, max {err.max():.4f} px")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("calibration", type=Path, nargs="?", default=Path("calibration.yml"),
                        help="omnicalib calibration.yml (Default: calibration.yml)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Ziel-NPZ (Default: <calibration>.npz)")
    parser.add_argument("-d", "--detections", type=Path, default=None,
                        help="omnicalib detections.pickle: Fit nur bis zum beobachteten "
                             "Einfallswinkel und Reprojektionspruefung gegen die Detektionen")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Bildgroesse in Pixeln; wird als image_size gespeichert und "
                             "begrenzt theta_max auf die Bildecken (mit --detections "
                             "wird sie sonst aus dem ersten Kalibrierbild gelesen)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Optionaler Kameraname, wird als 'camera' im NPZ gespeichert")
    parser.add_argument("--max-angle", type=float, default=None,
                        help="Maximaler Einfallswinkel des Fits in Grad, (0, 90). Default: "
                             "beobachteter Maximalwinkel mit --detections, sonst 89")
    args = parser.parse_args()

    if args.max_angle is not None and not 0.0 < args.max_angle < 90.0:
        parser.error("--max-angle muss im Bereich (0, 90) liegen")

    poly_theta, principal_point, extrinsics = load_calibration(args.calibration)
    if abs(poly_theta[0]) > 1e-9:
        print(f"Warnung: a0 = {poly_theta[0]:.3g} != 0, wird vom cv2-Modell ignoriert")

    views = None
    image_size = tuple(args.image_size) if args.image_size is not None else None
    if args.detections is not None:
        views = load_detections(args.detections, extrinsics)
        if image_size is None:
            image_size = image_size_from_file(views[0][0])
            if image_size is None:
                print(f"Hinweis: Bild {views[0][0]} nicht lesbar, Bildgroesse unbekannt")

    # Fit-Bereich bestimmen
    if args.max_angle is not None:
        theta_max, source = np.radians(args.max_angle), "--max-angle"
    elif views is not None:
        theta_max = max(incident_angle(wp @ R.T + T).max() for _, wp, _, R, T in views)
        source = "beobachtet (Detektionen)"
    else:
        theta_max, source = np.radians(89.0), "Default"
    print(f"theta_max {source}: {np.degrees(theta_max):.2f} Grad")

    if theta_max > THETA_LIMIT:
        print("Hinweis: cv2.fisheye kann nur theta < 90 Grad abbilden, begrenze auf 90 Grad")
        theta_max = THETA_LIMIT
    theta_mono = monotonic_limit(poly_theta)
    if theta_mono < theta_max:
        print(f"Hinweis: r(theta) ist nur bis {np.degrees(theta_mono):.1f} Grad monoton")
        theta_max = theta_mono
    if image_size is not None:
        W, H = image_size
        corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float64)
        r_corner = np.linalg.norm(corners - principal_point, axis=1).max()
        theta_corner = theta_for_radius(poly_theta, r_corner)
        if theta_corner < theta_max:
            print(f"Hinweis: Bildecke liegt bei {np.degrees(theta_corner):.1f} Grad")
            theta_max = theta_corner

    # Fit
    f, k, residual = fit_fisheye(poly_theta, theta_max)
    K = np.array([[f, 0.0, principal_point[0]],
                  [0.0, f, principal_point[1]],
                  [0.0, 0.0, 1.0]])
    D = k.reshape(4, 1)

    out = args.output or args.calibration.with_suffix(".npz")
    data = {"model": np.array("fisheye"), "K": K, "D": D,
            "theta_max": np.float64(theta_max)}
    if image_size is not None:
        data["image_size"] = np.array(image_size, dtype=np.int64)  # (W, H)
    if args.camera is not None:
        data["camera"] = np.array(args.camera)
    np.savez(out, **data)

    # Ausgabe und Pruefung
    np.set_printoptions(precision=6, suppress=True)
    print(f"Fit-Bereich: 0 .. {np.degrees(theta_max):.2f} Grad"
          + (f", Bildgroesse {image_size[0]}x{image_size[1]}" if image_size else ""))
    print(f"K =\n{K}")
    print(f"D = {D.ravel()}")
    print(f"Radiusfehler Fit:              {stats(np.abs(residual))}")
    print(f"cv2 vs. omnicalib (Zufall):    "
          f"{stats(verify_synthetic(poly_theta, principal_point, K, D, theta_max))}")
    if views is not None:
        err_omni, err_cv = verify_detections(views, poly_theta, principal_point, K, D)
        print(f"Reprojektion omnicalib-Modell: {stats(err_omni)}")
        print(f"Reprojektion cv2.fisheye:      {stats(err_cv)}"
              f"  ({len(views)} Bilder, {len(err_cv)} Punkte)")
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
