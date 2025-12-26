# sv-gmmubm

**sv-gmmubm** is a classical **speaker verification** system based on the  
**Gaussian Mixture Model – Universal Background Model (GMM-UBM)** framework.

The project implements a full text-independent speaker verification pipeline,
including UBM training, MAP adaptation, and log-likelihood ratio (LLR) scoring.

---

## Overview

The system follows a traditional speaker verification approach:

1. Feature extraction (VAD, MFCC, CMVN)
2. Universal Background Model (GMM-UBM) training
3. Baum–Welch sufficient statistics accumulation
4. MAP adaptation of speaker models
5. Log-likelihood ratio (LLR) scoring
6. Threshold-based accept/reject decision

This implementation is intended for academic and research purposes and is suitable
for classical datasets such as **TIMIT**.

---

## Feature Extraction

All acoustic feature extraction is performed using the **libvoicefeat** library,
which provides reusable implementations of:

- Voice Activity Detection (VAD)
- MFCC feature extraction
- Cepstral Mean and Variance Normalization (CMVN)

**sv-gmmubm does not reimplement feature extraction logic**, but relies on
`libvoicefeat` as a preprocessing module.

🔗 **libvoicefeat repository:**  
https://github.com/Serhii-Nesteruk/libvoicefeat

---

## Speaker Verification Framework

- **Model type:** Gaussian Mixture Models (diagonal covariance)
- **Speaker modeling:** MAP adaptation of UBM means
- **Scoring:** Log-likelihood ratio (LLR)
- **Mode:** Text-independent speaker verification

---

## Intended Use

This project is designed as:

- a reference implementation of a classical GMM-UBM speaker verification system
- a backend module for research or educational experiments
- a verification component that can be combined with different front-end
  feature extractors

---

## License

MIT
