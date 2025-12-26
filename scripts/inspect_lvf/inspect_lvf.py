#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

CEPS_TYPES = {0: "MFCC", 1: "LFCC", 2: "GFCC", 3: "PNCC", 4: "PLP"}
FILTERBANK_TYPES = {0: "Mel", 1: "Linear", 2: "Gammatone", 3: "Bark"}
MEL_SCALES = {0: "HTK", 1: "Slaney"}
COMPRESSION_TYPES = {0: "Log", 1: "PowerNormalized", 2: "CubeRoot"}

def read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]

def read_i32(f) -> int:
    return struct.unpack("<i", f.read(4))[0]

def read_f64(f) -> float:
    return struct.unpack("<d", f.read(8))[0]

def read_u8(f) -> int:
    return struct.unpack("<B", f.read(1))[0]

def inspect_lvf(path: Path, max_rows_print: int = 5, max_cols_print: int = 10, full_matrix: bool = False):
    with path.open("rb") as f:
        magic = f.read(8)
        version = read_u32(f)

        print(f"File: {path}")
        print(f"Magic: {magic!r}")
        print(f"Version: {version}")

        if magic != b"LVFEAT\x00\x00":
            raise RuntimeError("Bad magic. This doesn't look like an .lvf written by FeatureSerdes.")
        if version != 1:
            raise RuntimeError(f"Unsupported version: {version}")

        cep_u32 = read_u32(f)
        print(f"CepstralType: {cep_u32} ({CEPS_TYPES.get(cep_u32, 'UNKNOWN')})")

        sample_rate = read_i32(f)
        num_filters = read_i32(f)
        num_coeffs  = read_i32(f)
        min_freq = read_f64(f)
        max_freq = read_f64(f)
        include_energy = read_u8(f)
        filterbank = read_u32(f)
        mel_scale = read_u32(f)
        compression = read_u32(f)

        print("\nFeatureOptions:")
        print(f"  sampleRate: {sample_rate}")
        print(f"  numFilters: {num_filters}")
        print(f"  numCoeffs:  {num_coeffs}")
        print(f"  minFreq:    {min_freq}")
        print(f"  maxFreq:    {max_freq}")
        print(f"  includeEnergy: {bool(include_energy)}")
        print(f"  filterbank: {filterbank} ({FILTERBANK_TYPES.get(filterbank, 'UNKNOWN')})")
        print(f"  melScale:   {mel_scale} ({MEL_SCALES.get(mel_scale, 'UNKNOWN')})")
        print(f"  compressionType: {compression} ({COMPRESSION_TYPES.get(compression, 'UNKNOWN')})")

        rows = read_u32(f)
        cols = read_u32(f)
        print(f"\nFeatureMatrix: {rows} x {cols} (float32)")

        total = rows * cols
        data_bytes = f.read(total * 4)
        if len(data_bytes) != total * 4:
            raise RuntimeError("Unexpected EOF while reading matrix")

        floats = struct.unpack("<" + "f" * total, data_bytes)

        if full_matrix:
            print("\nMatrix (FULL):")
            for r in range(rows):
                row = floats[r*cols : (r+1)*cols]
                print("  " + " ".join(f"{v: .6f}" for v in row))
        else:
            print("\nMatrix preview (first rows/cols):")
            r_print = min(rows, max_rows_print)
            c_print = min(cols, max_cols_print)
            for r in range(r_print):
                row = floats[r*cols : r*cols + c_print]
                print(f"  row {r:>4}: " + " ".join(f"{v: .5f}" for v in row) + (" ..." if cols > c_print else ""))

        vad_n = read_u32(f)
        vad_bytes = f.read(vad_n)
        if len(vad_bytes) != vad_n:
            raise RuntimeError("Unexpected EOF while reading VAD flags")

        n_speech = sum(1 for b in vad_bytes if b == 1)
        n_nonspeech = vad_n - n_speech

        print(f"\nVADFlags: {vad_n} entries")
        print(f"  Speech:    {n_speech}")
        print(f"  NonSpeech: {n_nonspeech}")
        preview = " ".join(str(b) for b in vad_bytes[:50])
        print(f"  first 50:  {preview}" + (" ..." if vad_n > 50 else ""))

        extra = f.read(1)
        if extra:
            print("\n[WARN] File has extra bytes after expected end (format mismatch?)")
        else:
            print("\nOK: File structure matches expected .lvf format.")

def main():
    ap = argparse.ArgumentParser(description="Inspect .lvf files produced by sv::io::FeatureSerdes")
    ap.add_argument("lvf", type=Path, help="Path to .lvf file")
    ap.add_argument("--rows", type=int, default=5, help="How many matrix rows to print (preview mode)")
    ap.add_argument("--cols", type=int, default=10, help="How many cols per row to print (preview mode)")
    ap.add_argument("--full-matrix", action="store_true", help="Print the full matrix (all rows and columns)")
    args = ap.parse_args()

    inspect_lvf(args.lvf, args.rows, args.cols, args.full_matrix)

if __name__ == "__main__":
    main()
