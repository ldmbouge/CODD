import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def parse_matrix(text):
    """
    Converte una matrice testuale in un array NumPy.
    Ogni riga è separata da newline, gli elementi da spazi.
    """
    rows = text.strip().split("\n")
    matrix = []
    for r in rows:
        if r.strip() == "":
            continue
        nums = list(map(float, r.split()))
        matrix.append(nums)
    matrix = np.array(matrix)

    # Verifica simmetria
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        print("⚠ Warning: Distance matrix is not symmetric!")

    return matrix

def mds_with_error(D, n_components=2, random_state=0, verbose=True):
    """
    Applica MDS e calcola errori di ricostruzione.

    Returns:
        coords: coordinate 2D
        metrics: dizionario con metriche di errore
        D_reconstructed: matrice distanze ricostruite
    """
    D = np.asarray(D)
    n = len(D)

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress='auto'
    )
    coords = mds.fit_transform(D)

    # Distanze ricostruite
    diff = coords[:, None, :] - coords[None, :, :]
    D_reconstructed = np.sqrt(np.sum(diff**2, axis=2))

    # Maschera per elementi non-diagonali
    mask = ~np.eye(n, dtype=bool)

    # Errori assoluti
    abs_error = np.abs(D - D_reconstructed)

    # Errori relativi (solo per distanze non-zero)
    relative_error = np.zeros_like(abs_error)
    nonzero_mask = D > 1e-6
    relative_error[nonzero_mask] = abs_error[nonzero_mask] / D[nonzero_mask]

    # Metriche
    metrics = {
        'rms_error': np.sqrt(np.mean(abs_error[mask]**2)),
        'max_error': np.max(abs_error[mask]),
        'mean_error': np.mean(abs_error[mask]),
        'median_error': np.median(abs_error[mask]),
        'mean_relative_error': np.mean(relative_error[mask & nonzero_mask]) * 100,  # percentuale
        'max_relative_error': np.max(relative_error[mask & nonzero_mask]) * 100,
        'stress': mds.stress_,
    }

    if verbose:
        print("=" * 60)
        print("MDS RECONSTRUCTION QUALITY")
        print("=" * 60)
        print(f"Nodes: {n}")
        print(f"Stress: {metrics['stress']:.2f}")
        print(f"\nAbsolute Errors:")
        print(f"  RMS:    {metrics['rms_error']:6.2f}")
        print(f"  Mean:   {metrics['mean_error']:6.2f}")
        print(f"  Median: {metrics['median_error']:6.2f}")
        print(f"  Max:    {metrics['max_error']:6.2f}")
        print(f"\nRelative Errors:")
        print(f"  Mean:   {metrics['mean_relative_error']:6.2f}%")
        print(f"  Max:    {metrics['max_relative_error']:6.2f}%")

        # Interpretazione
        print(f"\n{'Assessment:'}")
        if metrics['rms_error'] < 5:
            print("  ✅ Excellent reconstruction - distances well preserved")
        elif metrics['rms_error'] < 15:
            print("  ✓ Good reconstruction - acceptable for most uses")
        else:
            print("  ⚠ Poor reconstruction - consider alternative approaches")

        print("=" * 60)

    return coords, metrics, D_reconstructed, abs_error

def plot_error_distribution(abs_error, D):
    """Visualizza la distribuzione degli errori."""
    mask = ~np.eye(len(D), dtype=bool)
    errors = abs_error[mask].flatten()

    plt.figure(figsize=(12, 4))

    # Istogramma errori
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=30, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(alpha=0.3)

    # Scatter: distanze originali vs ricostruite
    plt.subplot(1, 3, 2)
    from scipy.spatial.distance import squareform
    D_flat = D[mask]
    diff = abs_error[mask]
    D_recon_flat = D_flat - diff
    plt.scatter(D_flat, D_recon_flat, alpha=0.5, s=10)
    plt.plot([D_flat.min(), D_flat.max()],
             [D_flat.min(), D_flat.max()], 'r--', label='Perfect')
    plt.xlabel('Original Distance')
    plt.ylabel('Reconstructed Distance')
    plt.title('Distance Comparison')
    plt.legend()
    plt.grid(alpha=0.3)

    # Mappa di calore errori
    plt.subplot(1, 3, 3)
    plt.imshow(abs_error, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Absolute Error')
    plt.title('Error Heatmap')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.tight_layout()
    plt.show()

def plot_coordinates(coords):
    """Visualizza le coordinate 2D."""
    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=100, c='blue', alpha=0.6)

    # Etichetta nodi
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i), (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    # Evidenzia depot (nodo 0)
    plt.scatter(coords[0, 0], coords[0, 1], s=200, c='red',
               marker='s', label='Depot', zorder=5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('MDS 2D Projection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    text_matrix = """
0 96 1 97 93 48 93 64 54 85 38 31 95 46 100 37 19 58 44 98 66 55 87 54 71 5 97 16 86 58 77 89 98 18 99 79 55 40 59 3 89
96 0 95 54 96 48 37 32 57 54 58 93 67 106 89 62 80 120 89 63 36 76 59 47 86 91 93 80 68 114 104 25 46 78 53 20 62 113 81 95 7
1 95 0 96 92 48 92 64 53 84 38 30 94 45 99 37 20 57 43 97 65 54 86 54 70 5 96 15 85 57 76 88 97 17 98 79 54 39 58 4 88
97 54 96 0 46 68 17 60 43 12 75 77 14 81 36 79 94 92 65 9 40 48 12 74 45 94 41 82 20 84 62 29 9 81 2 63 42 91 49 98 52
93 96 92 46 0 88 61 88 54 43 91 64 32 57 13 95 98 60 50 38 68 39 37 98 22 92 6 82 28 53 24 71 55 83 48 98 49 69 34 96 93
48 48 48 68 88 0 56 16 34 58 10 54 72 72 89 14 32 87 57 73 28 52 63 10 68 43 88 33 67 82 84 49 65 31 69 31 40 74 58 47 41
93 37 92 17 61 56 0 45 40 18 65 77 30 86 52 69 86 99 69 26 29 52 24 61 55 89 57 77 33 91 73 12 9 75 16 46 42 95 55 93 35
64 32 64 60 88 16 45 0 37 51 26 66 66 82 85 30 48 97 66 66 20 56 57 16 71 59 87 49 63 91 88 36 54 47 60 15 43 86 62 63 25
54 57 53 43 54 34 40 37 0 31 37 38 42 49 55 41 51 63 32 45 21 19 34 44 34 51 54 39 35 57 51 41 44 38 45 50 6 57 25 55 50
85 54 84 12 43 58 18 51 31 0 65 65 15 70 36 69 82 82 54 15 31 37 6 65 37 82 40 70 15 74 55 29 16 69 14 58 30 80 38 86 50
38 58 38 75 91 10 65 26 37 65 0 48 78 66 92 4 22 81 54 80 36 52 69 16 69 33 91 24 72 78 84 59 73 22 77 41 42 67 58 37 51
31 93 30 77 64 54 77 66 38 65 48 0 70 18 72 50 43 33 14 76 57 29 65 63 42 32 68 27 61 30 46 79 80 29 79 81 35 20 31 34 86
95 67 94 14 32 72 30 66 42 15 78 70 0 72 22 82 93 81 57 6 46 41 9 80 34 93 27 81 9 73 49 42 23 80 16 73 40 83 40 97 65
46 106 45 81 57 72 86 82 49 70 66 18 72 0 68 68 61 15 17 78 70 34 69 81 39 49 63 45 63 12 35 89 86 47 83 97 45 12 32 49 99
100 89 99 36 13 89 52 85 55 36 92 72 22 68 0 96 104 73 58 27 65 45 30 98 30 99 7 89 22 65 37 64 45 89 38 93 50 80 41 103 86
37 62 37 79 95 14 69 30 41 69 4 50 82 68 96 0 20 83 57 84 40 56 73 17 73 32 95 25 76 80 87 62 77 23 81 43 46 68 61 35 55
19 80 20 94 98 32 86 48 51 82 22 43 93 61 104 20 0 76 53 96 57 59 85 36 76 15 101 16 85 73 86 80 93 15 95 63 54 58 64 17 73
58 120 57 92 60 87 99 97 63 82 81 33 81 15 73 83 76 0 31 87 84 47 80 96 47 62 66 60 72 8 36 103 98 62 94 112 59 18 44 61 113
44 89 43 65 50 57 69 66 32 54 54 14 57 17 58 57 53 31 0 63 53 17 53 67 28 45 54 37 48 25 33 72 70 39 67 81 28 26 17 47 82
98 63 97 9 38 73 26 66 45 15 80 76 6 78 27 84 96 87 63 0 46 47 11 80 40 96 33 84 15 79 55 38 18 83 11 71 43 89 46 100 61
66 36 65 40 68 28 29 20 21 31 36 57 46 70 65 40 57 84 53 46 0 40 37 34 53 62 67 50 43 78 71 23 37 48 41 30 26 77 45 66 29
55 76 54 48 39 52 52 56 19 37 52 29 41 34 45 56 59 47 17 47 40 0 36 62 17 54 42 44 32 39 32 56 53 44 50 69 14 43 6 58 69
87 59 86 12 37 63 24 57 34 6 69 65 9 69 30 73 85 80 53 11 37 36 0 71 33 85 34 73 9 72 51 34 19 72 14 64 32 79 37 89 56
54 47 54 74 98 10 61 16 44 65 16 63 80 81 98 17 36 96 67 80 34 62 71 0 78 49 98 40 76 92 94 52 70 38 75 27 50 83 68 52 40
71 86 70 45 22 68 55 71 34 37 69 42 34 39 30 73 76 47 28 40 53 17 33 78 0 70 26 60 25 39 18 63 52 61 47 83 28 51 12 74 80
5 91 5 94 92 43 89 59 51 82 33 32 93 49 99 32 15 62 45 96 62 54 85 49 70 0 96 12 84 61 78 85 95 14 96 74 53 44 58 4 84
97 93 96 41 6 88 57 87 54 40 91 68 27 63 7 95 101 66 54 33 67 42 34 98 26 96 0 86 25 59 30 68 50 86 43 96 49 75 38 100 90
16 80 15 82 82 33 77 49 39 70 24 27 81 45 89 25 16 60 37 84 50 44 73 40 60 12 86 0 72 57 70 73 83 2 84 64 41 43 48 16 73
86 68 85 20 28 67 33 63 35 15 72 61 9 63 22 76 85 72 48 15 43 32 9 76 25 84 25 72 0 64 42 43 28 71 22 71 31 74 31 88 65
58 114 57 84 53 82 91 91 57 74 78 30 73 12 65 80 73 8 25 79 78 39 72 92 39 61 59 57 64 0 29 95 90 59 86 106 52 20 36 61 107
77 104 76 62 24 84 73 88 51 55 84 46 49 35 37 87 86 36 33 55 71 32 51 94 18 78 30 70 42 29 0 81 70 71 64 101 45 47 26 80 98
89 25 88 29 71 49 12 36 41 29 59 79 42 89 64 62 80 103 72 38 23 56 34 52 63 85 68 73 43 95 81 0 21 71 28 34 44 98 60 89 23
98 46 97 9 55 65 9 54 44 16 73 80 23 86 45 77 93 98 70 18 37 53 19 70 52 95 50 83 28 90 70 21 0 81 7 55 45 96 54 99 44
18 78 17 81 83 31 75 47 38 69 22 29 80 47 89 23 15 62 39 83 48 44 72 38 61 14 86 2 71 59 71 71 81 0 83 62 40 45 49 18 71
99 53 98 2 48 69 16 60 45 14 77 79 16 83 38 81 95 94 67 11 41 50 14 75 47 96 43 84 22 86 64 28 7 83 0 62 44 93 51 100 51
79 20 79 63 98 31 46 15 50 58 41 81 73 97 93 43 63 112 81 71 30 69 64 27 83 74 96 64 71 106 101 34 55 62 62 0 56 101 75 78 14
55 62 54 42 49 40 42 43 6 30 42 35 40 45 50 46 54 59 28 43 26 14 32 50 28 53 49 41 31 52 45 44 45 40 44 56 0 54 19 57 55
40 113 39 91 69 74 95 86 57 80 67 20 83 12 80 68 58 18 26 89 77 43 79 83 51 44 75 43 74 20 47 98 96 45 93 101 54 0 43 43 106
59 81 58 49 34 58 55 62 25 38 58 31 40 32 41 61 64 44 17 46 45 6 37 68 12 58 38 48 31 36 26 60 54 49 51 75 19 43 0 62 74
3 95 4 98 96 47 93 63 55 86 37 34 97 49 103 35 17 61 47 100 66 58 89 52 74 4 100 16 88 61 80 89 99 18 100 78 57 43 62 0 88
89 7 88 52 93 41 35 25 50 50 51 86 65 99 86 55 73 113 82 61 29 69 56 40 80 84 90 73 65 107 98 23 44 71 51 14 55 106 74 88 0
"""

    D = parse_matrix(text_matrix)
    coords, metrics, D_reconstructed, abs_error = mds_with_error(D)

    # Visualizzazioni (opzionali)
    # plot_coordinates(coords)
    # plot_error_distribution(abs_error, D)
