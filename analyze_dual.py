#!/usr/bin/env python3
"""
Dual System Analyzer - Detects NOBO_F and NOBO_S + HawkEars
================================================================
"""

import os
import sys
import argparse
import subprocess
import time
import tempfile
from pathlib import Path
import pickle
import warnings

import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import scipy.signal

# Suppress PyTorch warnings specifically
torch.backends.mps.pin_memory = False

# Suppress progress bars and warnings
warnings.filterwarnings('ignore')
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
os.environ['TQDM_DISABLE'] = '1'
os.environ['OPENSOUNDSCAPE_QUIET'] = '1'

# NOBO classifier dependencies
try:
    import bioacoustics_model_zoo as bmz
    from opensoundscape.ml.shallow_classifier import MLPClassifier
    # Try to disable tqdm progress bars in bmz if possible
    try:
        import tqdm
        tqdm.tqdm.disable = True
    except:
        pass
except ImportError as e:
    pass


def filter_frequency_range(audio_data, sample_rate, low_freq=1300, high_freq=3000):
    """
    Filter audio data to only include frequencies between low_freq and high_freq
    """
    # Design bandpass filter
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are in valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    # Butterworth bandpass filter
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
    
    return filtered_audio


class NOBOClassifier:
    """Your NOBO classifier - properly detects NOBO_F and NOBO_S with frequency filtering"""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = model_path
        self.metadata_path = metadata_path or model_path.replace('.pth', '_metadata.pkl')
        self.classifier = None
        self.hawk_embedder = None
        self.optimal_thresholds = [0.5, 0.5]  # Default for both classes
        self.class_names = ['NOBO_F', 'NOBO_S']  # Your two classes
        self.is_loaded = self._load_classifier()
    
    def _load_classifier(self) -> bool:
        try:
            if not os.path.exists(self.model_path):
                return False
            
            # Load HawkEars embedder
            self.hawk_embedder = bmz.HawkEars()
            
            # Load classifier
            torch.serialization.add_safe_globals([MLPClassifier])
            self.classifier = torch.load(self.model_path, map_location='cpu', weights_only=False)
            self.classifier.eval()
            
            # Load metadata - handle both single and multi-label formats
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Check if it's multi-label format (new format)
                if 'optimal_thresholds' in metadata:
                    self.optimal_thresholds = metadata['optimal_thresholds']
                    # Map class indices to proper names: 0->NOBO_F, 1->NOBO_S
                    metadata_class_names = metadata.get('class_names', ['Class_0', 'Class_1'])
                    if metadata_class_names == ['Class_0', 'Class_1']:
                        self.class_names = ['NOBO_F', 'NOBO_S']  # Map to proper names
                    else:
                        self.class_names = metadata_class_names
                
                # Handle old single threshold format
                elif 'best_threshold' in metadata:
                    single_threshold = metadata['best_threshold']
                    self.optimal_thresholds = [single_threshold, single_threshold]
            
            return True
        except Exception as e:
            return False
    
    def analyze(self, audio_path: str) -> list:
        """Analyze audio for NOBO_F and NOBO_S detections with frequency filtering"""
        if not self.is_loaded:
            return []
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_path, sr=22050)
            duration = len(audio_data) / sample_rate
            
            # Apply frequency filtering (1000-4000 Hz)
            filtered_audio = filter_frequency_range(audio_data, sample_rate, 1000, 4000)
            
            detections = []
            segment_duration = 3.0
            segment_samples = int(segment_duration * sample_rate)
            hop_samples = int(1.5 * sample_rate)
            
            for start_sample in range(0, len(filtered_audio) - segment_samples + 1, hop_samples):
                end_sample = start_sample + segment_samples
                segment = filtered_audio[start_sample:end_sample]
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                # Get predictions for both classes
                predictions = self._predict_segment(segment, sample_rate)
                
                if predictions is not None:
                    # Check each class with its optimal threshold
                    for class_idx, (class_name, confidence) in enumerate(zip(self.class_names, predictions)):
                        threshold = self.optimal_thresholds[class_idx]
                        if confidence > threshold:
                            detections.append({
                                'start_time': round(start_time, 1),
                                'end_time': round(end_time, 1),
                                'species': class_name,  # NOBO_F or NOBO_S
                                'confidence': round(confidence, 3),
                                'source': 'nobo_classifier',
                                'threshold_used': round(threshold, 6)
                            })
            
            return detections
            
        except Exception as e:
            return []
    
    def _predict_segment(self, audio_segment: np.ndarray, sample_rate: int):
        """Predict NOBO_F and NOBO_S confidences for frequency-filtered segment"""
        try:
            # Ensure 3 seconds
            target_length = int(3.0 * sample_rate)
            if len(audio_segment) != target_length:
                if len(audio_segment) < target_length:
                    audio_segment = np.pad(audio_segment, (0, target_length - len(audio_segment)))
                else:
                    audio_segment = audio_segment[:target_length]
            
            # Save temp file for HawkEars (already frequency filtered)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_segment, sample_rate)
            
            try:
                # Get embeddings from filtered audio
                embeddings_raw = self.hawk_embedder.embed([temp_path])
                
                # Extract embedding vector (handle DataFrame properly)
                embedding_vector = self._extract_embedding_vector(embeddings_raw)
                
                if embedding_vector is None:
                    return None
                
                # Convert to tensor
                embedding_tensor = torch.FloatTensor(embedding_vector)
                if embedding_tensor.dim() == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                
                # Ensure 1920 features
                if embedding_tensor.shape[1] != 1920:
                    if embedding_tensor.shape[1] > 1920:
                        embedding_tensor = embedding_tensor[:, :1920]
                    else:
                        padding_size = 1920 - embedding_tensor.shape[1]
                        padding = torch.zeros(1, padding_size)
                        embedding_tensor = torch.cat([embedding_tensor, padding], dim=1)
                
                # Run classifier - returns [NOBO_F_confidence, NOBO_S_confidence]
                with torch.no_grad():
                    logits = self.classifier(embedding_tensor)  # Shape: [1, 2]
                    confidences = torch.sigmoid(logits).squeeze().numpy()  # Shape: [2]
                
                return confidences  # [NOBO_F_conf, NOBO_S_conf]
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            return None
    
    def _extract_embedding_vector(self, embeddings_raw):
        """Extract embedding vector from DataFrame/other formats"""
        try:
            # Handle pandas DataFrame (most common case)
            if isinstance(embeddings_raw, pd.DataFrame):
                embedding_vector = embeddings_raw.values
                if embedding_vector.ndim > 1:
                    embedding_vector = embedding_vector[0]
                return embedding_vector
            
            # Handle list of DataFrames
            elif isinstance(embeddings_raw, list) and len(embeddings_raw) > 0:
                first_item = embeddings_raw[0]
                if isinstance(first_item, pd.DataFrame):
                    embedding_vector = first_item.values
                    if embedding_vector.ndim > 1:
                        embedding_vector = embedding_vector[0]
                    return embedding_vector
                elif isinstance(first_item, np.ndarray):
                    return first_item if first_item.ndim == 1 else first_item[0]
                else:
                    return np.array(first_item)
            
            # Handle numpy array
            elif isinstance(embeddings_raw, np.ndarray):
                return embeddings_raw if embeddings_raw.ndim == 1 else embeddings_raw[0]
            
            # Try to convert
            else:
                return np.array(embeddings_raw)
                
        except Exception:
            return None


class DualSystemAnalyzer:
    """Combines original HawkEars with your NOBO_F/NOBO_S classifier"""
    
    def __init__(self, nobo_model_path: str = "data/nobo_multilabel_classifier.pth"):
        self.nobo_classifier = None
        
        # Initialize NOBO classifier
        if os.path.exists(nobo_model_path):
            self.nobo_classifier = NOBOClassifier(nobo_model_path)
    
    def analyze_directory(self, input_dir: str, output_dir: str):
        """Analyze directory with both systems"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_extensions = {'.wav', '.flac', '.mp3', '.WAV', '.FLAC', '.MP3'}
        audio_files = [f for f in input_path.rglob('*') if f.suffix in audio_extensions]
        
        if not audio_files:
            return
        
        total_hawkears = 0
        total_nobo_f = 0
        total_nobo_s = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            # Step 1: Run original HawkEars (full spectrum)
            hawkears_detections = self._run_original_hawkears(audio_file, output_path)
            
            # Step 2: Run your NOBO classifier (1000-4000 Hz filtered)
            nobo_detections = []
            if self.nobo_classifier:
                nobo_detections = self.nobo_classifier.analyze(str(audio_file))
            
            # Step 3: Combine and save unified output
            combined_detections = hawkears_detections + nobo_detections
            combined_detections.sort(key=lambda x: x['start_time'])
            
            self._save_combined_labels(combined_detections, audio_file, output_path)
            
            # Count detections by type
            total_hawkears += len(hawkears_detections)
            nobo_f_count = len([d for d in nobo_detections if d['species'] == 'NOBO_F'])
            nobo_s_count = len([d for d in nobo_detections if d['species'] == 'NOBO_S'])
            total_nobo_f += nobo_f_count
            total_nobo_s += nobo_s_count
        
        print(f"FINAL RESULTS:")
        print(f"   Files processed: {len(audio_files)}")
        print(f"   HawkEars detections (full spectrum): {total_hawkears}")
        print(f"   NOBO_F detections (1000-4000Hz): {total_nobo_f}")
        print(f"   NOBO_S detections (1000-4000Hz): {total_nobo_s}")
        print(f"   Total detections: {total_hawkears + total_nobo_f + total_nobo_s}")
        print(f"   Output directory: {output_path}")
    
    def _run_original_hawkears(self, audio_file: Path, output_dir: Path) -> list:
        """Run original HawkEars analyze.py and parse results"""
        try:
            # Create temp output directory
            temp_output = output_dir / "temp_hawkears"
            temp_output.mkdir(exist_ok=True)
            
            # Run original analyze.py
            result = subprocess.run([
                sys.executable, 'analyze.py',
                '-i', str(audio_file),
                '-o', str(temp_output)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return []
            
            # Parse HawkEars output
            hawkears_file = temp_output / f"{audio_file.stem}_HawkEars.txt"
            detections = []
            
            if hawkears_file.exists():
                with open(hawkears_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                parts = line.split('\t')
                                if len(parts) >= 3:
                                    start_time = float(parts[0])
                                    end_time = float(parts[1])
                                    
                                    # Parse species and confidence
                                    label = parts[2]
                                    if ';' in label:
                                        species, conf_str = label.split(';', 1)
                                        confidence = float(conf_str)
                                    else:
                                        species = label
                                        confidence = 0.5
                                    
                                    detections.append({
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'species': species,
                                        'confidence': confidence,
                                        'source': 'hawkears'
                                    })
                            except (ValueError, IndexError):
                                continue
                
                # Clean up temp file
                hawkears_file.unlink()
            
            return detections
            
        except subprocess.TimeoutExpired:
            return []
        except Exception as e:
            return []
    
    def _save_combined_labels(self, detections: list, audio_file: Path, output_dir: Path):
        """Save combined detections to unified label file"""
        output_file = output_dir / f"{audio_file.stem}_combined.txt"
        
        with open(output_file, 'w') as f:
            f.write("# Format: start_time\\tend_time\\tspecies;confidence_source\n")
            
            if detections:
                for det in detections:
                    start_time = det['start_time']
                    end_time = det['end_time']
                    species = det['species']
                    confidence = det['confidence']
                    source = det['source']
                    
                    # Add threshold info for NOBO detections
                    if 'threshold_used' in det:
                        threshold_info = f"_t{det['threshold_used']:.3f}"
                    else:
                        threshold_info = ""
                    
                    # Format: time_start time_end species;confidence_source
                    f.write(f"{start_time:.2f}\t{end_time:.2f}\t{species};{confidence:.3f}_{source}{threshold_info}\n")
            else:
                f.write("# No detections found\n")


def main():
    parser = argparse.ArgumentParser(description='Dual System: HawkEars + NOBO_F/NOBO_S (1000-4000Hz)')
    parser.add_argument('-i', '--input', required=True, help='Input audio file or directory')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--nobo-model', default='data/nobo_multilabel_classifier.pth', help='NOBO model path')
    parser.add_argument('--override-thresholds', nargs=2, type=float, metavar=('NOBO_F', 'NOBO_S'), 
                       help='Override optimal thresholds [NOBO_F_threshold NOBO_S_threshold]')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent if input_path.is_file() else input_path
    
    # Initialize dual analyzer
    analyzer = DualSystemAnalyzer(args.nobo_model)
    
    # Override NOBO thresholds if specified
    if args.override_thresholds and analyzer.nobo_classifier:
        analyzer.nobo_classifier.optimal_thresholds = args.override_thresholds
    
    # Process input
    if input_path.is_file():
        # Single file - treat as directory with one file
        analyzer.analyze_directory(str(input_path.parent), str(output_dir))
    else:
        # Directory
        analyzer.analyze_directory(str(input_path), str(output_dir))


if __name__ == '__main__':
    main()