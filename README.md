# Quantum-Kernel Hybrid RAG

**A retrieval-augmented generation pipeline that reranks classical embedding retrieval results using a quantum kernel (PennyLane + Qiskit), benchmarked against classical-only retrieval on a small support-FAQ corpus.**

---

## Overview

This project builds a small RAG pipeline over a customer-support FAQ corpus (refunds, shipping, warranty, payments, etc.) and augments classical embedding retrieval (`gemini-embedding-001` + cosine similarity) with a **quantum kernel reranking stage** implemented in PennyLane, optionally executable on real IBM Quantum hardware via Qiskit Runtime.

The goal was to investigate whether a quantum kernel — computed via a fidelity-based feature map circuit — can meaningfully reorder or improve retrieval results beyond what classical cosine similarity already provides.

**Short answer, stated honestly up front:** on this corpus, it doesn't improve retrieval accuracy, but a carefully-blended version doesn't hurt it either. See [Findings](#findings) for the full result and why that's still a useful conclusion.

---

## Architecture

```
Query
  │
  ▼
Classical retrieval (Gemini embeddings + cosine similarity, top-k=3)
  │
  ▼
Quantum kernel reranking (PennyLane fidelity kernel, reorder-only, top-k=3)
  combined_score = alpha * classical_score + (1 - alpha) * quantum_score
  │
  ▼
Context assembly (all top-k candidates, reordered — none dropped)
  │
  ▼
Gemini generates final answer from context
```

**Key design choice: reranking is reorder-only, never filtering.** An earlier version filtered the reranked set down to fewer documents than were retrieved classically, and this silently dropped the correct answer from context in 2 of 3 test queries when the quantum kernel disagreed with classical ranking. Setting the reranker's `top_k` equal to the retriever's `top_k` eliminates this failure mode structurally — the quantum stage can only reorder candidates, so the final LLM synthesis step always sees every classically-retrieved document.

---

## The Quantum Kernel

A fidelity-based quantum kernel: two documents are encoded via a parameterized feature map (`RY`/`RZ` rotations + a `CNOT` entangling layer), and their similarity is the probability of measuring the all-zeros state after applying one feature map followed by the adjoint of the other:

```python
def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))
```

Since `gemini-embedding-001` produces 768+ dimensional vectors and the kernel only uses `n_qubits` (8, in the reported results) rotation angles, embeddings are compressed via chunk-averaging and rescaled to span `[-π, π]` before being fed to the circuit.

---

## Bugs Found & Fixed

Documenting these because diagnosing them was most of the actual engineering work.

1. **Angle saturation (silent, no error thrown).** `gemini-embedding-001` components are small (~0.01–0.05 in magnitude). Fed directly into `RY`/`RZ` without rescaling, every rotation was near-identity regardless of input — the kernel returned ≈1.0 for *every* pair of documents, giving zero discrimination. Fixed by rescaling compressed embeddings to `[-π, π]`.
2. **IBM Open Plan session restriction.** `qml.device("qiskit.remote", ...)` creates a Qiskit `Session` internally. IBM's Open (free) plan does not permit session-mode execution — only job/batch mode — so any real-hardware run failed with a 400 error. Running on real hardware requires bypassing PennyLane's device abstraction and calling `SamplerV2` directly in job mode against a manually transpiled circuit.
3. **Filter-mode reranking dropped correct answers.** Setting the reranker's `top_k` below the retriever's `top_k` meant a classically-correct top document could be reranked out of the final context entirely — this caused the pipeline to answer "I don't know" to a question the corpus explicitly answered. Fixed by making reranking reorder-only (see Architecture above).
4. **Qubit-count rescaling artifact.** Quantum kernel scores are not comparable across different `n_qubits` settings — `probs[0]` (the all-zeros measurement probability) mechanically shrinks as the Hilbert space grows (2⁸ vs 2¹⁰ basis states), independent of document similarity. All reported results below use a fixed `n_qubits=8`.
5. **Redundant embedding calls.** Query and document embeddings were being re-fetched from the API on every call rather than cached, causing avoidable rate-limit errors during experimentation. Fixed with an in-memory cache for query embeddings and reuse of the corpus's precomputed document embeddings during reranking.

---

## Findings

### 1. Alpha sensitivity (blend weight between classical and quantum scores)

`combined_score = alpha * classical_score + (1 - alpha) * quantum_score`

| Query | a=0.0 | a=0.2 | a=0.4 | a=0.5 | a=0.6 | a=0.8 | a=1.0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| What is your refund policy? | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Do you offer bulk discounts? | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| How can I contact support? | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

(✓ = combined top-1 matches classical top-1)

Pure classical weighting (alpha=1.0) is the only setting with 100% agreement across this query set. The quantum term measurably shifts top-1 ranking even at alpha=0.8, because in cases where classical scores are closely clustered (e.g. 0.723 vs 0.694), even a 20%-weighted quantum contribution is enough to flip the ordering.

### 2. Ground-truth accuracy test (lexical-confusion cases)

Designed to test whether quantum reranking corrects classical retrieval mistakes, using query/document pairs with deliberate surface-level lexical overlap (e.g. "additional fee" appearing in both the correct document and a plausible distractor).

| Query | a=0.0 | a=0.2 | a=0.5 | a=0.8 | a=1.0 |
|---|:---:|:---:|:---:|:---:|:---:|
| Gift wrapping fee? | ✗ | ✓ | ✓ | ✓ | ✓ |
| Extended coverage cost? | ✗ | ✗ | ✓ | ✓ | ✓ |
| Installment payment charge? | ✓ | ✓ | ✓ | ✓ | ✓ |

**Classical top-1 was already correct on all 3 test cases at baseline** (no reranking). This means the experiment could not test whether the quantum kernel *corrects* a classical error, because classical retrieval didn't make one — `gemini-embedding-001` was robust to the surface-lexical-overlap confusers used here. What it does show: **quantum-dominant reranking (low alpha) actively introduces errors on 2 of 3 cases**, requiring alpha ≥ 0.5 to fully recover correctness.

### Conclusion

On this corpus and task, classical embedding retrieval was already highly accurate, and no test case was found where quantum kernel reranking corrected a classical retrieval error. A conservative, classical-dominant blend (alpha ≥ 0.8) avoids the regressions that quantum-dominant reranking introduces, but does not demonstrate a positive contribution from the quantum stage on this data. A more adversarial test corpus — one specifically constructed to defeat classical embedding similarity (rather than relying on lexical overlap alone) — would be needed to properly evaluate whether this kernel design adds value beyond classical retrieval.

---

## Setup

```bash
pip install google-genai pennylane pennylane-qiskit qiskit-ibm-runtime python-dotenv numpy
```

Create a `.env` file with:
```
GOOGLE_API_KEY=your_key_here
```

Run:
```bash
python hybrid_quantum_rag.py
```

By default this runs on the local `default.qubit` simulator. Real IBM hardware execution requires an IBM Quantum account and is currently only supported via job-mode execution due to Open Plan session restrictions (see [Bugs Found & Fixed](#bugs-found--fixed), item 2) — the relevant code path is commented in the script with notes on what would need to change.

## Limitations

- Small corpus (15 documents) — findings here should not be assumed to generalize to larger-scale retrieval.
- Embedding compression (768+ dims → 8 via chunk-averaging) is lossy; a more principled dimensionality reduction (e.g. PCA fit on the corpus) was not tested.
- Real-hardware execution is untested end-to-end due to the Open Plan session restriction; only simulator results are reported here.
