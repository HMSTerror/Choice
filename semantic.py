"""Training-only semantic preprocessing for PreferDiff and DDSR.

DDSR uses the OPQ option in Appendix B of arXiv:2410.23994.
The supplied fixed text embeddings are reused; no validation/test targets fit
the whitening transform or semantic codebooks.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def whiten_features(features, seen, width=None):
    """Fit on training items; preserve all source dimensions outside smoke runs."""
    import numpy as np
    import torch
    fit = features[seen]
    mean = fit.mean(0)
    if width is not None:
        rank = min(width, len(fit) - 1, fit.shape[1])
        _, singular, vectors = torch.pca_lowrank(fit - mean, q=rank, center=False, niter=4)
        scale = (len(fit) - 1) ** .5 / singular.clamp_min(1e-5)
        return torch.nn.functional.pad((features - mean) @ vectors * scale, (0, width - rank))
    centered = (fit - mean).double()
    covariance = centered.T @ centered / (len(fit) - 1)
    # NumPy/LAPACK avoids MKL SSYEVD workspace errors observed at native 3072D.
    eigenvalues, vectors = np.linalg.eigh(covariance.cpu().numpy())
    eigenvalues = torch.from_numpy(eigenvalues).to(features)
    vectors = torch.from_numpy(vectors).to(features)
    return (features - mean) @ vectors * eigenvalues.clamp_min(1e-6).rsqrt()


def semantic_preprocessing(source_root, output, config, settings):
    import numpy as np
    import pandas as pd
    import torch
    from data import DATASETS, load_split, metadata, write_json
    output = Path(output)
    target = output / "semantic.pt"
    if target.exists():
        return
    started = time.perf_counter()
    directory = Path(source_root) / DATASETS[config["dataset"]]
    filename = settings["embedding_files"][config["dataset"]]
    features = torch.as_tensor(np.asarray(pd.read_pickle(directory / filename)), dtype=torch.float32)
    n = metadata(output)["item_count"]
    if features.ndim != 2 or len(features) != n or not torch.isfinite(features).all():
        raise ValueError(f"Expected one finite embedding per item in {directory / filename}")
    training = load_split(output, "train")
    seen = torch.unique(torch.cat((training["seq"].flatten(), training["target"])))
    seen = seen[seen < n]
    # Preserve native semantic width for the PreferDiff paper's fixed features.
    torch.manual_seed(config["seed"])
    width = config["hidden_size"] if config["smoke"] else features.shape[1]
    projected = whiten_features(features, seen, width if config["smoke"] else None)
    m, k = (2, 16) if config["smoke"] else (settings["codebooks"], settings["codebook_size"])
    if features.shape[1] % m or k != 256 and not config["smoke"]:
        raise ValueError("DDSR OPQ requires divisible feature width and 256-entry codebooks")
    if config["smoke"]:
        # Small PQ codebooks keep the protocol test independent of full OPQ fits.
        chunks = features.reshape(n, m, -1)
        books, codes = [], []
        generator = torch.Generator().manual_seed(config["seed"])
        k = min(k, len(seen))
        for level in range(m):
            x = chunks[seen, level]
            centroids = x[torch.randperm(len(x), generator=generator)[:k]].clone()
            for _ in range(5):
                assignment = torch.cdist(x, centroids).argmin(-1)
                for c in range(k):
                    if (assignment == c).any():
                        centroids[c] = x[assignment == c].mean(0)
            books.append(centroids)
            codes.append(torch.cdist(chunks[:, level], centroids).argmin(-1))
        books, codes = torch.stack(books), torch.stack(codes, -1)
    else:
        # Faiss and PyTorch can load conflicting OpenMP runtimes. Keep native
        # quantization in a process that imports Faiss/NumPy, never PyTorch.
        with tempfile.TemporaryDirectory(prefix="opq_", dir=output) as temporary:
            temporary = Path(temporary)
            np.savez(temporary / "input.npz", features=features.numpy(), seen=seen.numpy())
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(temporary),
                            str(m), str(config["seed"]), str(config["threads"]), str(settings["opq_iterations"])], check=True)
            with np.load(temporary / "output.npz") as quantized:
                codes = torch.from_numpy(quantized["codes"].astype(np.int64))
                books = torch.from_numpy(quantized["centroids"].copy())
    # A dedicated nonrankable padding semantic ID.
    codes = torch.cat((codes, torch.full((1, m), k, dtype=torch.long)))
    torch.save({"features": projected, "codes": codes, "centroids": books,
                "codebooks": m, "codebook_size": k}, target)
    write_json(output / "semantic.json", {"embedding_file": str(directory / filename),
               "items": n, "native_width": features.shape[1], "preferdiff_width": width,
               "codebooks": m, "codebook_size": k, "fit_split": "train",
               "fit_items": len(seen), "preferdiff_whitening": "PCA-smoke" if config["smoke"] else "full-native-width",
               "quantization": "PQ-smoke" if config["smoke"] else "OPQ",
               "elapsed_seconds": time.perf_counter() - started})


def quantize_worker(directory, m, seed, threads, iterations):
    """Internal subprocess entry: a single native OpenMP runtime."""
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(threads)
    with np.load(directory / "input.npz") as values:
        raw, seen = np.ascontiguousarray(values["features"]), values["seen"]
    opq = faiss.OPQMatrix(raw.shape[1], m)
    opq.niter, opq.verbose = iterations, True
    opq.train(np.ascontiguousarray(raw[seen]))
    rotated = np.ascontiguousarray(opq.apply_py(raw))
    quantizer = faiss.ProductQuantizer(rotated.shape[1], m, 8)
    quantizer.cp.seed = seed
    quantizer.train(np.ascontiguousarray(rotated[seen]))
    np.savez(directory / "output.npz", codes=quantizer.compute_codes(rotated),
             centroids=faiss.vector_to_array(quantizer.centroids).reshape(m, 256, -1))


if __name__ == "__main__":
    directory, *numbers = sys.argv[1:]
    quantize_worker(Path(directory), *map(int, numbers))
