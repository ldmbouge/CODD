#!/usr/bin/env python3
"""
TSPTW to Solomon VRPTW converter.

Input:
  - TSPTW instance in the format:
      Line 1: number of nodes N
      Next N lines: distance matrix (NxN)
      Next N lines: time windows [ready_time, due_date] for each node

Output:
  - Solomon style VRPTW instance:
      <Instance name>

      VEHICLE
      NUMBER     CAPACITY
        K           Q

      CUSTOMER
      CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME

          0     ...
          1     ...
        ...

Customer 0 is the depot.
"""

import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import sys
import os


def read_tsptw_instance(filename):
    """
    Read TSPTW instance file.

    Format:
    - Line 1: number of nodes
    - Next N lines: distance matrix
    - Next N lines: time windows [ready_time, due_date]

    Returns:
        n: number of nodes
        dist_matrix: NxN distance matrix
        time_windows: list of (ready, due) tuples
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Number of nodes
    n = int(lines[0])

    # Distance matrix
    dist_matrix = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        dist_matrix.append(row)
    dist_matrix = np.array(dist_matrix)

    # Time windows
    time_windows = []
    for i in range(n + 1, 2 * n + 1):
        parts = lines[i].split()
        ready = int(parts[0])
        due = int(parts[1])
        time_windows.append((ready, due))

    return n, dist_matrix, time_windows


def mds_transform(dist_matrix, random_state=42, max_iter=3000, n_init=10, eps=1e-9):
    """
    Transform distance matrix to x-y coordinates using MDS.

    Args:
        dist_matrix: NxN distance matrix
        random_state: random seed
        max_iter: maximum iterations for optimization (default: 3000, higher = more precise)
        n_init: number of random initializations (default: 10, higher = better chance of global optimum)
        eps: convergence threshold (default: 1e-9, lower = more precise)

    Returns:
        coords: Nx2 array of coordinates
        metrics: dictionary with reconstruction quality metrics
        reconstructed_dist: reconstructed distance matrix
    """
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
        max_iter=max_iter,
        n_init=n_init,
        eps=eps,
    )

    coords = mds.fit_transform(dist_matrix)

    # Reconstruction error
    D_reconstructed = squareform(pdist(coords, "euclidean"))
    n = len(dist_matrix)
    mask = ~np.eye(n, dtype=bool)

    abs_error = np.abs(dist_matrix - D_reconstructed)

    metrics = {
        "rms_error": np.sqrt(np.mean(abs_error[mask] ** 2)),
        "mean_error": np.mean(abs_error[mask]),
        "max_error": np.max(abs_error[mask]),
        "stress": mds.stress_,
    }

    return coords, metrics, D_reconstructed

def make_coords_positive(coords):
    """
    Shift coordinates so that the minimum x and y are positive.
    Does not change relative distances.
    """
    min_x = np.min(coords[:, 0])
    min_y = np.min(coords[:, 1])

    shift_x = -min_x if min_x < 0 else 0
    shift_y = -min_y if min_y < 0 else 0

    return coords + np.array([shift_x, shift_y])


def print_distance_matrix(matrix, title="Distance Matrix"):
    """
    Print a distance matrix in a formatted way.
    """
    n = len(matrix)
    print(f"\n{title}")
    print("=" * 70)

    # Print header
    print("     ", end="")
    for j in range(n):
        print(f"{j:8d}", end="")
    print()
    print("-" * (5 + n * 8))

    # Print rows
    for i in range(n):
        print(f"{i:3d}: ", end="")
        for j in range(n):
            print(f"{matrix[i, j]:8.2f}", end="")
        print()
    print()


def write_solomon_vrptw_instance(
    filename,
    instance_name,
    n,
    coords_int,
    time_windows,
    num_vehicles=1,
    vehicle_capacity=10000,
    service_time=10,
    scale_coords=1.0,
    default_demand=1,
):
    """
    Write a Solomon style VRPTW instance.

    Layout:

    <Instance name>

    VEHICLE
    NUMBER     CAPACITY
      K           Q

    CUSTOMER
    CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME

          0   x0 y0 ...
          1   x1 y1 ...
        ...

    Args:
        coords_int: Already scaled and rounded integer coordinates (Nx2)
        scale_coords: Additional scaling (default 1.0 = no scaling)
    """

    # Apply any additional scaling if needed
    if scale_coords != 1.0:
        coords_int = np.rint(coords_int * scale_coords).astype(int)

    with open(filename, "w") as f:
        # Instance name
        f.write(f"{instance_name}\n")
        f.write("\n")

        # Vehicle block
        f.write("VEHICLE\n")
        f.write("NUMBER     CAPACITY\n")
        f.write(f"  {num_vehicles:<9d}{vehicle_capacity}\n")
        f.write("\n")

        # Customer block
        f.write("CUSTOMER\n")
        f.write("CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME\n")
        f.write("\n")

        for i in range(n):
            x = coords_int[i, 0]
            y = coords_int[i, 1]
            ready, due = time_windows[i]

            demand = 0 if i == 0 else default_demand
            service = 0 if i == 0 else service_time

            # Use whitespace separated columns, like Solomon instances
            line = (
                f"{i:7d}"
                f"{x:9d}"
                f"{y:11d}"
                f"{demand:11d}"
                f"{ready:12d}"
                f"{due:10d}"
                f"{service:13d}\n"
            )
            f.write(line)


def convert_tsptw_to_vrptw_solomon(
    input_file,
    output_file=None,
    instance_name=None,
    num_vehicles=1,
    vehicle_capacity=10000,
    service_time=10,
    scale_coords=100.0,
    default_demand=1,
    random_state=42,
    print_matrices=True,
    mds_max_iter=30000,
    mds_n_init=10,
    mds_eps=1e-19,
):
    """
    Convert a TSPTW instance to a Solomon style VRPTW instance.
    """

    # Output file
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_solomon_vrptw.txt"

    # Instance name
    if instance_name is None:
        instance_name = os.path.splitext(os.path.basename(input_file))[0]

    print("=" * 70)
    print("TSPTW to Solomon VRPTW Converter")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Step 1: read TSPTW
    print("Step 1: Reading TSPTW instance...")
    n, dist_matrix, time_windows = read_tsptw_instance(input_file)
    print(f"  Loaded {n} nodes")
    print(f"  Time window range: [{time_windows[0][0]}, {max(tw[1] for tw in time_windows)}]")

    if print_matrices:
        print_distance_matrix(dist_matrix, "Original Distance Matrix")
    print()

    # Step 2: MDS to coordinates
    print("Step 2: Applying MDS transformation...")
    print(f"  Using max_iter={mds_max_iter}, n_init={mds_n_init}, eps={mds_eps}")
    coords, metrics, mds_reconstructed_dist = mds_transform(
        dist_matrix,
        random_state=random_state,
        max_iter=mds_max_iter,
        n_init=mds_n_init,
        eps=mds_eps
    )
    print("  MDS completed")
    print(f"  Stress: {metrics['stress']:.2f}")
    print(f"  RMS Error (MDS): {metrics['rms_error']:.2f}")
    print(f"  Mean Error (MDS): {metrics['mean_error']:.2f}")
    print(f"  Max Error (MDS): {metrics['max_error']:.2f}")

    if metrics["rms_error"] < 5:
        quality = "Excellent"
    elif metrics["rms_error"] < 15:
        quality = "Good"
    else:
        quality = "Poor (distances are hard to embed in 2D)"
    print(f"  Quality: {quality}")
    print()

    # Step 2b: Scale and round coordinates (what actually gets written to file)
    print("Step 2b: Scaling and rounding coordinates...")
    coords_positive = make_coords_positive(coords)
    coords_scaled = coords_positive * scale_coords
    coords_int = np.rint(coords_scaled).astype(int)

    # Reconstruct distance matrix from the ROUNDED integer coordinates
    final_reconstructed_dist = squareform(pdist(coords_int, 'euclidean'))

    # Calculate error metrics for the final rounded coordinates
    n = len(dist_matrix)
    mask = ~np.eye(n, dtype=bool)
    abs_error_final = np.abs(dist_matrix - final_reconstructed_dist)

    final_metrics = {
        "rms_error": np.sqrt(np.mean(abs_error_final[mask] ** 2)),
        "mean_error": np.mean(abs_error_final[mask]),
        "max_error": np.max(abs_error_final[mask]),
    }

    print(f"  Coordinates scaled by {scale_coords} and rounded to integers")
    print(f"  RMS Error (final rounded): {final_metrics['rms_error']:.2f}")
    print(f"  Mean Error (final rounded): {final_metrics['mean_error']:.2f}")
    print(f"  Max Error (final rounded): {final_metrics['max_error']:.2f}")

    if print_matrices:
        print_distance_matrix(mds_reconstructed_dist, "Reconstructed Distance Matrix (from MDS float coordinates)")
        print_distance_matrix(final_reconstructed_dist, "Final Reconstructed Distance Matrix (from rounded integer coordinates)")

        # Print error matrices
        error_matrix_mds = np.abs(dist_matrix - mds_reconstructed_dist)
        print_distance_matrix(error_matrix_mds, "Absolute Error Matrix (MDS)")

        error_matrix_final = np.abs(dist_matrix - final_reconstructed_dist)
        print_distance_matrix(error_matrix_final, "Absolute Error Matrix (Final Rounded)")
    print()

    # Step 3: write Solomon VRPTW
    print("Step 3: Writing Solomon VRPTW instance...")
    write_solomon_vrptw_instance(
        output_file,
        instance_name,
        n,
        coords_int,  # Use the already-rounded integer coordinates
        time_windows,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        service_time=service_time,
        scale_coords=1.0,  # No additional scaling needed - already done
        default_demand=default_demand,
    )
    print(f"  VRPTW instance written to: {output_file}")
    print()

    print("=" * 70)
    print("Conversion completed successfully")
    print("=" * 70)

    return {
        "n": n,
        "coords": coords,
        "coords_int": coords_int,
        "mds_metrics": metrics,
        "final_metrics": final_metrics,
        "mds_reconstructed_dist": mds_reconstructed_dist,
        "final_reconstructed_dist": final_reconstructed_dist,
        "output_file": output_file,
    }


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python tsptw_to_vrptw_solomon.py <input_tsptw_file> [options]")
        print()
        print("Options:")
        print("  -o, --output FILE       Output VRPTW file (default: auto generated)")
        print("  -n, --name NAME         Instance name (default: from filename)")
        print("  -k, --vehicles K        Number of vehicles (default: 1)")
        print("  -q, --capacity Q        Vehicle capacity (default: 10000)")
        print("  -s, --service-time N    Service time for customers (default: 0)")
        print("  -d, --demand N          Default demand per customer (default: 1)")
        print("  --scale FACTOR          Coordinate scaling factor (default: 100.0)")
        print("  --seed N                Random seed for MDS (default: 42)")
        print("  --no-print-matrices     Don't print distance matrices")
        print("  --mds-max-iter N        Max MDS iterations (default: 3000, higher = more precise)")
        print("  --mds-n-init N          Number of MDS initializations (default: 10)")
        print("  --mds-eps FLOAT         MDS convergence threshold (default: 1e-9)")
        print()
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = None
    instance_name = None
    num_vehicles = 1
    vehicle_capacity = 10000
    service_time = 0
    default_demand = 1
    scale_coords = 1.0
    random_state = 42
    print_matrices = True
    mds_max_iter = 3000
    mds_n_init = 10
    mds_eps = 1e-9

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg in ["-o", "--output"] and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg in ["-n", "--name"] and i + 1 < len(sys.argv):
            instance_name = sys.argv[i + 1]
            i += 2
        elif arg in ["-k", "--vehicles"] and i + 1 < len(sys.argv):
            num_vehicles = int(sys.argv[i + 1])
            i += 2
        elif arg in ["-q", "--capacity"] and i + 1 < len(sys.argv):
            vehicle_capacity = int(sys.argv[i + 1])
            i += 2
        elif arg in ["-s", "--service-time"] and i + 1 < len(sys.argv):
            service_time = int(sys.argv[i + 1])
            i += 2
        elif arg in ["-d", "--demand"] and i + 1 < len(sys.argv):
            default_demand = int(sys.argv[i + 1])
            i += 2
        elif arg == "--scale" and i + 1 < len(sys.argv):
            scale_coords = float(sys.argv[i + 1])
            i += 2
        elif arg == "--seed" and i + 1 < len(sys.argv):
            random_state = int(sys.argv[i + 1])
            i += 2
        elif arg == "--no-print-matrices":
            print_matrices = False
            i += 1
        elif arg == "--mds-max-iter" and i + 1 < len(sys.argv):
            mds_max_iter = int(sys.argv[i + 1])
            i += 2
        elif arg == "--mds-n-init" and i + 1 < len(sys.argv):
            mds_n_init = int(sys.argv[i + 1])
            i += 2
        elif arg == "--mds-eps" and i + 1 < len(sys.argv):
            mds_eps = float(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown option: {arg}")
            sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    try:
        convert_tsptw_to_vrptw_solomon(
            input_file,
            output_file=output_file,
            instance_name=instance_name,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            service_time=service_time,
            scale_coords=scale_coords,
            default_demand=default_demand,
            random_state=random_state,
            print_matrices=print_matrices,
            mds_max_iter=mds_max_iter,
            mds_n_init=mds_n_init,
            mds_eps=mds_eps,
        )
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
