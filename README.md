# sv-gmmubm

This project implements a classical text-independent speaker verification system based on the GMM-UBM approach. It follows a standard pipeline including acoustic feature extraction, training of a universal background model, MAP adaptation of speaker models, and log-likelihood ratio scoring. The implementation is intended primarily for educational and experimental use.

Acoustic features such as VAD, MFCC, and CMVN are extracted using the external libvoicefeat library (https://github.com/Serhii-Nesteruk/libvoicefeat), while this repository focuses on speaker modeling and scoring. The system can be applied to common speech datasets (e.g. TIMIT) and serves as a reference implementation of a traditional GMM-UBM speaker verification setup.
