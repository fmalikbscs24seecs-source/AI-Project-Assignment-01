import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


DATASET_PATH = r"C:\Users\MZ\Downloads\Dataset"



class WaveformAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.real_files = []
        self.fake_files = []

    def find_audio_files(self):
        """Find and categorize audio files"""
        print("🔍 Searching for audio files...")

        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']

        for ext in audio_extensions:
            for file_path in Path(self.data_path).rglob(ext):
                filename = file_path.name.lower()
                file_path_str = str(file_path)

                # Categorize based on filename or folder name
                if any(keyword in filename or keyword in file_path_str.lower()
                       for keyword in ['real', 'genuine', 'original', 'natural']):
                    self.real_files.append(file_path)
                elif any(keyword in filename or keyword in file_path_str.lower()
                         for keyword in ['fake', 'synthetic', 'generated', 'artificial']):
                    self.fake_files.append(file_path)
                else:
                    # If unknown, distribute evenly
                    if len(self.real_files) <= len(self.fake_files):
                        self.real_files.append(file_path)
                    else:
                        self.fake_files.append(file_path)

        print(f" Found {len(self.real_files)} real audio files")
        print(f" Found {len(self.fake_files)} fake audio files")

        if not self.real_files and not self.fake_files:
            print(" No audio files found! Check dataset path.")
            return False
        return True

    def create_waveform_comparison(self):
        """Create waveform comparison plots"""
        print("🌊 Creating waveform analysis...")

        # Create output folder
        output_dir = Path(self.data_path) / "plots"
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Select examples
        real_example = self.real_files[0] if self.real_files else None
        fake_example = self.fake_files[0] if self.fake_files else None

        # --- PLOT 1: Combined Waveform ---
        if real_example and fake_example:
            try:
                audio_real, sr_real = librosa.load(real_example, sr=None)
                audio_fake, sr_fake = librosa.load(fake_example, sr=None)

                time_real = np.arange(len(audio_real)) / sr_real
                time_fake = np.arange(len(audio_fake)) / sr_fake

                axes[0, 0].plot(time_real, audio_real, color='#2E8B57',
                                label='Real Sound', linewidth=1.5, alpha=0.8)
                axes[0, 0].plot(time_fake, audio_fake, color='#DC143C',
                                label='Fake Sound', linewidth=1.5, alpha=0.7)
                axes[0, 0].set_title("A) Real vs Fake Waveform", fontsize=14, fontweight='bold')
                axes[0, 0].set_xlabel("Time (s)")
                axes[0, 0].set_ylabel("Amplitude")
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
            except Exception as e:
                print(f" Error plotting combined waveform: {e}")

        # --- PLOT 2: Real Waveform ---
        if real_example:
            try:
                audio_real, sr_real = librosa.load(real_example, sr=None)
                max_samples = min(len(audio_real), 5 * sr_real)
                time_real = np.arange(max_samples) / sr_real
                axes[0, 1].plot(time_real, audio_real[:max_samples], color='#2E8B57', linewidth=2)
                axes[0, 1].set_title("B) Real Sound (First 5 sec)", fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel("Time (s)")
                axes[0, 1].set_ylabel("Amplitude")
                axes[0, 1].grid(alpha=0.3)
            except Exception as e:
                print(f" Error in real waveform plot: {e}")

        # --- PLOT 3: Fake Waveform ---
        if fake_example:
            try:
                audio_fake, sr_fake = librosa.load(fake_example, sr=None)
                max_samples = min(len(audio_fake), 5 * sr_fake)
                time_fake = np.arange(max_samples) / sr_fake
                axes[1, 0].plot(time_fake, audio_fake[:max_samples], color='#DC143C', linewidth=2)
                axes[1, 0].set_title("C) Fake Sound (First 5 sec)", fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel("Time (s)")
                axes[1, 0].set_ylabel("Amplitude")
                axes[1, 0].grid(alpha=0.3)
            except Exception as e:
                print(f" Error in fake waveform plot: {e}")

        # --- PLOT 4: Summary ---
        axes[1, 1].axis("off")
        summary = "Waveform analysis complete.\n\n"
        if real_example:
            summary += f"Real: {os.path.basename(real_example)}\n"
        if fake_example:
            summary += f"Fake: {os.path.basename(fake_example)}"
        axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                        fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        pdf_path = output_dir / "waveform_analysis.pdf"
        plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f" Waveform PDF saved successfully at:\n{pdf_path}")

    def create_detailed_waveform_analysis(self):
        """Optional: Zoomed and histogram analysis"""
        if not self.real_files or not self.fake_files:
            print(" Skipping detailed analysis (need both real & fake files).")
            return

        output_dir = Path(self.data_path) / "plots"
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        try:
            real_audio, sr_real = librosa.load(self.real_files[0], sr=None)
            fake_audio, sr_fake = librosa.load(self.fake_files[0], sr=None)

            target_sr = min(sr_real, sr_fake, 22050)
            if sr_real != target_sr:
                real_audio = librosa.resample(real_audio, orig_sr=sr_real, target_sr=target_sr)
            if sr_fake != target_sr:
                fake_audio = librosa.resample(fake_audio, orig_sr=sr_fake, target_sr=target_sr)

            time = np.arange(len(real_audio)) / target_sr
            axes[0].plot(time, real_audio, color='#2E8B57', label='Real', alpha=0.7)
            axes[0].plot(time, fake_audio, color='#DC143C', label='Fake', alpha=0.7)
            axes[0].set_title("Waveform Overlay", fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            zoom_dur = 2
            zoom_samples = int(zoom_dur * target_sr)
            axes[1].plot(np.arange(zoom_samples)/target_sr, real_audio[:zoom_samples], color='#2E8B57', label='Real')
            axes[1].plot(np.arange(zoom_samples)/target_sr, fake_audio[:zoom_samples], color='#DC143C', label='Fake')
            axes[1].set_title("Zoomed (First 2 sec)", fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            axes[2].hist(real_audio, bins=50, alpha=0.7, density=True, label='Real', color='#2E8B57')
            axes[2].hist(fake_audio, bins=50, alpha=0.7, density=True, label='Fake', color='#DC143C')
            axes[2].set_title("Amplitude Distribution", fontweight='bold')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            pdf_path = output_dir / "detailed_waveform_analysis.pdf"
            plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f" Detailed waveform PDF saved successfully at:\n{pdf_path}")
        except Exception as e:
            print(f" Error in detailed waveform analysis: {e}")


def main():
    print("=== ASSIGNMENT 1 - WAVEFORM ANALYSIS ===")
    analyzer = WaveformAnalyzer(DATASET_PATH)
    if not analyzer.find_audio_files():
        return
    analyzer.create_waveform_comparison()
    analyzer.create_detailed_waveform_analysis()
    print("\n All waveform analyses completed successfully!")


if __name__ == "__main__":
    main()
