#!/usr/bin/env python3
"""
Script to combine multiple WAV audio files into a single output file.
Supports both pydub (recommended) and wave (standard library) methods.
"""

import argparse
import os
import sys


def combine_wav_files_pydub(input_files, output_file, target_sample_rate=None, target_channels=None):
    """
    Combine multiple WAV files using pydub (requires: pip install pydub)
    This method handles different sample rates and channels automatically.
    Files with different parameters will be normalized to match the first file,
    or to specified target parameters.

    Args:
        input_files: List of input WAV file paths
        output_file: Output WAV file path
        target_sample_rate: Target sample rate (Hz). If None, uses first file's rate.
        target_channels: Target channels (1=mono, 2=stereo). If None, uses first file's channels.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Error: pydub is not installed. Install it with: pip install pydub")
        print("Alternatively, use the --use-wave flag to use the standard library.")
        sys.exit(1)

    # Load the first audio file
    combined = AudioSegment.from_wav(input_files[0])
    print(f"Loading: {input_files[0]} (sample_rate={combined.frame_rate}Hz, channels={combined.channels})")

    # Set target parameters if specified, otherwise use first file's parameters
    if target_sample_rate is None:
        target_sample_rate = combined.frame_rate
    if target_channels is None:
        target_channels = combined.channels

    # Normalize first file to target parameters if needed
    if combined.frame_rate != target_sample_rate or combined.channels != target_channels:
        print(f"Normalizing first file to {target_sample_rate}Hz, {target_channels} channel(s)")
        combined = combined.set_frame_rate(target_sample_rate).set_channels(target_channels)

    # Append all other audio files
    for audio_file in input_files[1:]:
        print(f"Adding: {audio_file}")
        audio = AudioSegment.from_wav(audio_file)
        print(f"  Original: sample_rate={audio.frame_rate}Hz, channels={audio.channels}")

        # Normalize to target parameters
        if audio.frame_rate != target_sample_rate or audio.channels != target_channels:
            print(f"  Normalizing to {target_sample_rate}Hz, {target_channels} channel(s)")
            audio = audio.set_frame_rate(target_sample_rate).set_channels(target_channels)

        combined += audio

    # Export the combined audio
    combined.export(output_file, format="wav")
    print(f"\nSuccessfully combined {len(input_files)} files into: {output_file}")
    print(f"Output: sample_rate={combined.frame_rate}Hz, channels={combined.channels}")
    print(f"Output duration: {len(combined) / 1000.0:.2f} seconds")


def combine_wav_files_wave(input_files, output_file, allow_different_params=False):
    """
    Combine multiple WAV files using the standard library wave module.

    Note: By default, this requires all files to have the same parameters.
    If files have different parameters, use pydub method (default) instead,
    or set allow_different_params=True to attempt conversion (requires pydub).

    Args:
        input_files: List of input WAV file paths
        output_file: Output WAV file path
        allow_different_params: If True, convert files with different parameters using pydub
    """
    import wave

    # Open the first file to get parameters
    with wave.open(input_files[0], "rb") as first_wav:
        params = first_wav.getparams()
        frames = first_wav.readframes(first_wav.getnframes())
        print(f"First file parameters: {params}")

    # Open output file for writing
    with wave.open(output_file, "wb") as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

        # Append frames from remaining files
        skipped_count = 0
        for audio_file in input_files[1:]:
            print(f"Adding: {audio_file}")
            with wave.open(audio_file, "rb") as in_wav:
                in_params = in_wav.getparams()
                # Verify parameters match
                if in_params != params:
                    if allow_different_params:
                        # Try to convert using pydub
                        try:
                            from pydub import AudioSegment

                            print(f"  Converting {audio_file} to match parameters...")
                            # Load with pydub, normalize, save to temp file
                            import tempfile

                            audio = AudioSegment.from_wav(audio_file)
                            # Convert to match first file's parameters
                            audio = audio.set_frame_rate(params.framerate).set_channels(params.nchannels)
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp_path = tmp.name
                                audio.export(tmp_path, format="wav")
                            # Read converted file
                            with wave.open(tmp_path, "rb") as converted_wav:
                                converted_frames = converted_wav.readframes(converted_wav.getnframes())
                                out_wav.writeframes(converted_frames)
                            os.unlink(tmp_path)
                            print("  Successfully converted and added")
                        except ImportError:
                            print("  Error: pydub not available for conversion. Skipping.")
                            print(
                                "  Install pydub or use default method (without --use-wave) to combine files with different parameters."
                            )
                            skipped_count += 1
                            continue
                        except Exception as e:
                            print(f"  Error converting file: {e}. Skipping.")
                            skipped_count += 1
                            continue
                    else:
                        print(f"  Warning: {audio_file} has different parameters ({in_params}). Skipping.")
                        print("  Use default method (without --use-wave) or --allow-different-params to handle this.")
                        skipped_count += 1
                        continue
                else:
                    frames = in_wav.readframes(in_wav.getnframes())
                    out_wav.writeframes(frames)

    if skipped_count > 0:
        print(f"\nWarning: {skipped_count} file(s) were skipped due to parameter mismatches.")
    print(f"\nSuccessfully combined {len(input_files) - skipped_count} files into: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple WAV audio files into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all audio_*.wav files in current directory (handles different parameters automatically)
  python combine_audio.py -o combined.wav audio_*.wav

  # Combine specific files with different parameters (normalized automatically)
  python combine_audio.py -o output.wav audio_0.wav audio_1.wav audio_2.wav

  # Combine files and normalize to specific sample rate and channels
  python combine_audio.py -o output.wav --target-sample-rate 44100 --target-channels 2 audio_*.wav

  # Use wave module (requires matching parameters, or use --allow-different-params)
  python combine_audio.py -o output.wav --use-wave audio_*.wav

  # Use wave module with automatic conversion for different parameters
  python combine_audio.py -o output.wav --use-wave --allow-different-params audio_*.wav
        """,
    )

    parser.add_argument(
        "input_files", nargs="+", help="Input WAV files to combine (can use wildcards like audio_*.wav)"
    )

    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")

    parser.add_argument(
        "--use-wave",
        action="store_true",
        help="Use standard library wave module instead of pydub (requires matching audio parameters)",
    )

    parser.add_argument(
        "--allow-different-params",
        action="store_true",
        help="Allow combining files with different parameters (only works with --use-wave, requires pydub for conversion)",
    )

    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Target sample rate in Hz (default: use first file's rate)",
    )

    parser.add_argument(
        "--target-channels",
        type=int,
        choices=[1, 2],
        default=None,
        help="Target number of channels: 1=mono, 2=stereo (default: use first file's channels)",
    )

    parser.add_argument("--sort", action="store_true", help="Sort input files alphabetically before combining")

    args = parser.parse_args()

    # Expand wildcards and get absolute paths
    input_files = []
    for pattern in args.input_files:
        if "*" in pattern or "?" in pattern:
            import glob

            input_files.extend(sorted(glob.glob(pattern)))
        else:
            input_files.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    input_files = [f for f in input_files if not (f in seen or seen.add(f))]

    # Sort if requested
    if args.sort:
        input_files = sorted(input_files)

    # Validate files exist
    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)
        if not f.lower().endswith(".wav"):
            print(f"Warning: {f} does not have .wav extension")

    if len(input_files) == 0:
        print("Error: No input files found")
        sys.exit(1)

    print(f"Combining {len(input_files)} audio files...")
    print(f"Input files: {', '.join(input_files)}")

    # Combine using selected method
    if args.use_wave:
        combine_wav_files_wave(input_files, args.output, allow_different_params=args.allow_different_params)
    else:
        combine_wav_files_pydub(
            input_files, args.output, target_sample_rate=args.target_sample_rate, target_channels=args.target_channels
        )


if __name__ == "__main__":
    main()
    # python combine_audio.py -o output.wav --use-wave --allow-different-params audio_*.wav
