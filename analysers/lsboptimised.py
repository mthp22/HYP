from collections import defaultdict
import os
import sys
import math
import struct
import random
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LSBAnalyzer")

def read_bmp_pixels(file_path):
    """
    Read pixel data from a BMP file.
    Returns a dictionary with image information and pixel data.
    """
    try:
        with open(file_path, 'rb') as f:
            # Read BMP header
            header = f.read(14)
            if len(header) < 14:
                return {'error': 'Invalid BMP file (header too small)'}
            
            # Check BMP signature
            if header[0:2] != b'BM':
                return {'error': 'Not a valid BMP file (wrong signature)'}
            
            # Get file size and data offset from header
            file_size = struct.unpack('<I', header[2:6])[0]
            data_offset = struct.unpack('<I', header[10:14])[0]
            
            # Read DIB header (BITMAPINFOHEADER)
            dib_header = f.read(40)
            if len(dib_header) < 40:
                return {'error': 'Invalid BMP file (DIB header too small)'}
            
            # Extract image dimensions and bit depth
            width = struct.unpack('<i', dib_header[4:8])[0]
            height = struct.unpack('<i', dib_header[8:12])[0]
            bit_depth = struct.unpack('<H', dib_header[14:16])[0]
            compression = struct.unpack('<I', dib_header[16:20])[0]
            
            # Validate dimensions and bit depth
            if width <= 0 or height == 0:
                return {'error': f'Invalid dimensions: {width}x{height}'}
            
            if bit_depth not in [24, 32]:
                return {'error': f'Unsupported bit depth: {bit_depth}. Only 24 and 32-bit BMPs are supported.'}
            
            if compression != 0:
                return {'error': f'Compressed BMP files not supported (compression type: {compression})'}
            
            # Calculate row size (including padding)
            bytes_per_pixel = bit_depth // 8
            row_size = ((width * bit_depth + 31) // 32) * 4
            
            # Read pixel data
            is_bottom_up = height > 0
            if not is_bottom_up:
                height = abs(height)  # Handle top-down BMPs
            
            # Seek to the beginning of pixel data
            f.seek(data_offset)
            
            # Initialize pixel array
            pixels = []
            
            # Read all pixel data
            for y in range(height):
                row = []
                row_data = f.read(row_size)
                
                if len(row_data) < row_size:
                    return {'error': f'Truncated pixel data at row {y}'}
                
                for x in range(width):
                    pixel_offset = x * bytes_per_pixel
                    if pixel_offset + bytes_per_pixel > len(row_data):
                        return {'error': f'Pixel data out of bounds at row {y}, column {x}'}
                    
                    # BGR format in BMP
                    b = row_data[pixel_offset]
                    g = row_data[pixel_offset + 1]
                    r = row_data[pixel_offset + 2]
                    
                    row.append((r, g, b))
                
                if is_bottom_up:
                    # BMP stores rows bottom-to-top by default
                    pixels.insert(0, row)
                else:
                    # Top-down BMP
                    pixels.append(row)
            
            # Flatten the pixels if needed for analysis
            flat_pixels = []
            for row in pixels:
                for pixel in row:
                    flat_pixels.append(pixel)
            
            return {
                'width': width,
                'height': height,
                'bit_depth': bit_depth,
                'pixels': flat_pixels,
                'success': True
            }
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {'error': f'File not found: {file_path}'}
    except MemoryError:
        logger.error(f"Memory error while processing {file_path}")
        return {'error': 'Not enough memory to process the image'}
    except Exception as e:
        logger.error(f"Error reading BMP file {file_path}: {str(e)}")
        return {'error': f'Error reading BMP file: {str(e)}'}

def extract_lsb_plane(pixel_data):
    """Extract LSB planes from pixel data"""
    if not pixel_data or 'pixels' not in pixel_data or not pixel_data.get('success', False):
        return {'error': 'Invalid pixel data'}
    
    pixels = pixel_data['pixels']
    num_pixels = len(pixels)
    
    red_lsbs = []
    green_lsbs = []
    blue_lsbs = []
    combined_lsbs = []

    for r, g, b in pixels:
        r_lsb = r & 1
        g_lsb = g & 1
        b_lsb = b & 1
        
        red_lsbs.append(r_lsb)
        green_lsbs.append(g_lsb)
        blue_lsbs.append(b_lsb)
        combined_lsbs.extend([r_lsb, g_lsb, b_lsb])
    
    return {
        'red_lsbs': red_lsbs,
        'green_lsbs': green_lsbs,
        'blue_lsbs': blue_lsbs,
        'combined_lsbs': combined_lsbs,
        'success': True
    }

def calculate_entropy(bit_sequence):
    """Calculate Shannon entropy of a bit sequence"""
    if not bit_sequence:
        return 0.0
    
    total = len(bit_sequence)
    count_1 = sum(bit_sequence)
    count_0 = total - count_1
    
    if count_0 == 0 or count_1 == 0:
        return 0.0
    
    p0 = count_0 / total
    p1 = count_1 / total
    
    try:
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        return entropy
    except (ValueError, ZeroDivisionError):
        return 0.0
    
    return entropy

def chi_square_test(bit_sequence):
    """Perform chi-square test for randomness"""
    if not bit_sequence:
        return 0.0, 'low'
    
    total = len(bit_sequence)
    count_1 = sum(bit_sequence)
    count_0 = total - count_1
    
    expected = total / 2.0
    chi_square = ((count_0 - expected) ** 2 + (count_1 - expected) ** 2) / expected
    
    # Simple p-value approximation
    if chi_square < 0.004:
        return chi_square, 'very_low'
    elif chi_square < 3.841:
        return chi_square, 'low'  # below 95% confidence
    else:
        return chi_square, 'high'  # above 95% confidence
    
    return chi_square, 'unknown'

def runs_test(bit_sequence):
    """Wald-Wolfowitz runs test"""
    if not bit_sequence or len(bit_sequence) < 2:
        return 0.0, 'unknown'
    
    n = len(bit_sequence)
    n1 = sum(bit_sequence)
    n0 = n - n1
    
    if n1 == 0 or n0 == 0:
        return 0.0, 'nonrandom'
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if bit_sequence[i] != bit_sequence[i-1]:
            runs += 1
    
    # Expected runs and variance
    expected_runs = (2 * n1 * n0) / n + 1
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    
    if variance <= 0:
        return 0.0, 'unknown'
    
    # Z-score
    z_score = abs(runs - expected_runs) / math.sqrt(variance)
    
    # Convert to p-value (simplified)
    if z_score > 1.96:
        return z_score, 'nonrandom'  # Non-random at 95% confidence
    else:
        return z_score, 'random'  # Likely random
    
    return z_score, 'unknown'

def analyze_pairs_of_values(pixel_data):
    """Pairs of Values (PoV) attack for LSB steganography detection"""
    if not pixel_data or 'pixels' not in pixel_data or not pixel_data.get('success', False):
        return {'error': 'Invalid pixel data'}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        even_values = defaultdict(int)
        odd_values = defaultdict(int)
        
        for pixel in pixels:
            value = pixel[channel_idx]
            if value % 2 == 0:
                even_values[value] += 1
            else:
                odd_values[value - 1] += 1
        
        # Calculate chi-square for pairs
        chi_square_sum = 0
        sample_count = 0
        degrees_freedom = 0
        
        for value in range(0, 256, 2):
            even_count = even_values[value]
            odd_count = odd_values[value]
            
            pair_sum = even_count + odd_count
            if pair_sum > 0:
                expected = pair_sum / 2
                chi_square_sum += ((even_count - expected)**2 + (odd_count - expected)**2) / expected
                sample_count += pair_sum
                degrees_freedom += 1
        
        # Higher chi-square values indicate potential steganography
        results[channel_name] = {
            'chi_square': chi_square_sum,
            'degrees_freedom': degrees_freedom,
            'sample_count': sample_count,
            'stego_likelihood': 'high' if chi_square_sum > degrees_freedom * 3 else 'low'
        }
    
    return results

def analyze_sample_pairs(pixel_data):
    """Sample Pairs Analysis for detecting LSB embedding"""
    if not pixel_data or 'pixels' not in pixel_data or not pixel_data.get('success', False):
        return {'error': 'Invalid pixel data'}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        # Extract channel values
        values = [pixel[channel_idx] for pixel in pixels]
        
        # Count same-value and different-value adjacent pairs
        same_pairs = 0
        different_pairs = 0
        total_pairs = len(values) - 1
        
        for i in range(total_pairs):
            val1 = values[i]
            val2 = values[i + 1]
            
            # Compare LSB
            if val1 % 2 == val2 % 2:
                same_pairs += 1
            else:
                different_pairs += 1
        
        # Calculate ratio of same to different pairs
        # In natural images, should be close to 1.0
        # In stego images, often closer to 0.5
        ratio = same_pairs / different_pairs if different_pairs > 0 else float('inf')
        
        # Estimate embedding rate based on ratio
        estimated_embedding = 0.0
        if 0.4 < ratio < 0.6:
            estimated_embedding = 0.75  # High embedding
        elif 0.3 < ratio < 0.7:
            estimated_embedding = 0.5   # Medium embedding
        elif 0.2 < ratio < 0.8:
            estimated_embedding = 0.25  # Low embedding
        
        results[channel_name] = {
            'same_pairs': same_pairs,
            'different_pairs': different_pairs,
            'ratio': ratio,
            'estimated_embedding': estimated_embedding,
            'stego_likelihood': 'high' if estimated_embedding > 0.4 else 'low'
        }
    
    return results

def analyze_histogram_center_of_mass(pixel_data):
    """Analyze center of mass of histograms"""
    if not pixel_data or 'pixels' not in pixel_data or not pixel_data.get('success', False):
        return {'error': 'Invalid pixel data'}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        # Create histogram
        histogram = [0] * 256
        for pixel in pixels:
            value = pixel[channel_idx]
            if 0 <= value < 256:
                histogram[value] += 1
        
        # Calculate center of mass
        total_mass = sum(histogram)
        weighted_sum = sum(i * count for i, count in enumerate(histogram))
        
        center_of_mass = weighted_sum / total_mass if total_mass > 0 else 0
        
        # Calculate even/odd distribution
        even_sum = sum(histogram[i] for i in range(0, 256, 2))
        odd_sum = sum(histogram[i] for i in range(1, 256, 2))
        
        even_odd_ratio = even_sum / odd_sum if odd_sum > 0 else float('inf')
        
        # Natural images typically have even_odd_ratio close to 1.0
        # Significant deviation suggests potential steganography
        results[channel_name] = {
            'center_of_mass': center_of_mass,
            'even_odd_ratio': even_odd_ratio,
            'stego_likelihood': 'high' if abs(even_odd_ratio - 1.0) > 0.1 else 'low'
        }
    
    return results

def analyze_bit_plane_complexity(pixel_data):
    """Analyze complexity of bit planes"""
    if not pixel_data or 'pixels' not in pixel_data or not pixel_data.get('success', False):
        return {'error': 'Invalid pixel data'}
    
    pixels = pixel_data['pixels']
    results = {}
    
    # Define a function to calculate bit plane complexity
    def calculate_complexity(bit_plane):
        if not bit_plane or len(bit_plane) < 2:
            return 0.0
        
        # Count transitions (changes from 0->1 or 1->0)
        transitions = sum(1 for i in range(1, len(bit_plane)) if bit_plane[i] != bit_plane[i-1])
        max_transitions = len(bit_plane) - 1
        
        # Normalize to range [0,1]
        return transitions / max_transitions if max_transitions > 0 else 0
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        # Extract bit planes (LSB to MSB)
        bit_planes = [[] for _ in range(8)]
        
        for pixel in pixels:
            value = pixel[channel_idx]
            for bit_idx in range(8):
                bit_planes[bit_idx].append((value >> bit_idx) & 1)
        
        # Calculate complexity for each bit plane
        complexities = [calculate_complexity(plane) for plane in bit_planes]
        
        # LSB plane in stego images often has higher complexity than natural images
        lsb_complexity = complexities[0]
        avg_higher_complexity = sum(complexities[1:]) / 7 if len(complexities) > 1 else 0
        
        # Compare LSB complexity to higher bit planes
        complexity_ratio = lsb_complexity / avg_higher_complexity if avg_higher_complexity > 0 else float('inf')
        
        # In steganographic images, LSB complexity is often higher than other bit planes
        results[channel_name] = {
            'lsb_complexity': lsb_complexity,
            'avg_higher_complexity': avg_higher_complexity,
            'complexity_ratio': complexity_ratio,
            'stego_likelihood': 'high' if complexity_ratio > 1.1 else 'low'
        }
    
    return results

def detect_stego(file_path, quick=False):
    """
    Detect steganography in a BMP image using multiple LSB steganalysis methods.
    Returns (is_stego, confidence, details)
    """
    # Read pixels from BMP file
    pixel_data = read_bmp_pixels(file_path)
    if 'error' in pixel_data:
        logger.error(f"Error reading BMP file: {pixel_data['error']}")
        return False, 0.0, {'error': pixel_data['error']}
    
    # Extract LSB planes
    lsb_data = extract_lsb_plane(pixel_data)
    if 'error' in lsb_data:
        logger.error(f"Error extracting LSB plane: {lsb_data['error']}")
        return False, 0.0, {'error': lsb_data['error']}
    
    # Initialize suspicion score (0-7, higher means more suspicious)
    suspicion_score = 0
    details = {}
    
    # Calculate entropy for each channel's LSB plane
    red_entropy = calculate_entropy(lsb_data['red_lsbs'])
    green_entropy = calculate_entropy(lsb_data['green_lsbs'])
    blue_entropy = calculate_entropy(lsb_data['blue_lsbs'])
    combined_entropy = calculate_entropy(lsb_data['combined_lsbs'])
    
    # Natural images typically have LSB entropy < 0.98
    # Higher entropy suggests potential steganography
    details['entropy'] = {
        'red': red_entropy,
        'green': green_entropy,
        'blue': blue_entropy,
        'combined': combined_entropy
    }
    
    # Adjust suspicion score based on entropy
    if combined_entropy > 0.999:  # Very high entropy
        suspicion_score += 2
    elif combined_entropy > 0.995:  # High entropy
        suspicion_score += 1
    
    # Run chi-square test
    chi_red, chi_red_result = chi_square_test(lsb_data['red_lsbs'])
    chi_green, chi_green_result = chi_square_test(lsb_data['green_lsbs'])
    chi_blue, chi_blue_result = chi_square_test(lsb_data['blue_lsbs'])
    chi_combined, chi_combined_result = chi_square_test(lsb_data['combined_lsbs'])
    
    details['chi_square'] = {
        'red': {'value': chi_red, 'result': chi_red_result},
        'green': {'value': chi_green, 'result': chi_green_result},
        'blue': {'value': chi_blue, 'result': chi_blue_result},
        'combined': {'value': chi_combined, 'result': chi_combined_result}
    }
    
    # Adjust suspicion score based on chi-square results
    if chi_combined_result == 'high' and chi_combined > 10.0:
        suspicion_score += 1
    
    # Run runs test
    runs_red, runs_red_result = runs_test(lsb_data['red_lsbs'])
    runs_green, runs_green_result = runs_test(lsb_data['green_lsbs'])
    runs_blue, runs_blue_result = runs_test(lsb_data['blue_lsbs'])
    runs_combined, runs_combined_result = runs_test(lsb_data['combined_lsbs'])
    
    details['runs_test'] = {
        'red': {'value': runs_red, 'result': runs_red_result},
        'green': {'value': runs_green, 'result': runs_green_result},
        'blue': {'value': runs_blue, 'result': runs_blue_result},
        'combined': {'value': runs_combined, 'result': runs_combined_result}
    }
    
    # Adjust suspicion score based on runs test results
    if runs_combined_result == 'random' and runs_combined > 2.5:
        suspicion_score += 1  # Random runs often indicate steganography
    
    # Skip advanced tests in quick mode
    if not quick:
        # Pairs of Values analysis
        pov_results = analyze_pairs_of_values(pixel_data)
        details['pairs_of_values'] = pov_results
        
        # Count high likelihood channels
        pov_high_count = sum(1 for channel, data in pov_results.items() 
                          if isinstance(data, dict) and data.get('stego_likelihood') == 'high')
        
        if pov_high_count >= 3:
            suspicion_score += 1
        
        # Sample Pairs Analysis
        spa_results = analyze_sample_pairs(pixel_data)
        details['sample_pairs'] = spa_results
        
        spa_high_count = sum(1 for channel, data in spa_results.items()
                           if isinstance(data, dict) and data.get('stego_likelihood') == 'high')
        
        if spa_high_count >= 3:
            suspicion_score += 1
        
        # Histogram analysis
        hist_results = analyze_histogram_center_of_mass(pixel_data)
        details['histogram'] = hist_results
        
        hist_high_count = sum(1 for channel, data in hist_results.items()
                            if isinstance(data, dict) and data.get('stego_likelihood') == 'high')
        
        if hist_high_count >= 3:
            suspicion_score += 0.5
        
        # Bit plane complexity analysis
        complexity_results = analyze_bit_plane_complexity(pixel_data)
        details['bit_plane_complexity'] = complexity_results
        
        complexity_high_count = sum(1 for channel, data in complexity_results.items()
                                 if isinstance(data, dict) and data.get('stego_likelihood') == 'high')
        
        if complexity_high_count >= 3:
            suspicion_score += 0.5
    
    # Calculate final results
    is_stego = suspicion_score >= 4
    
    # Map suspicion score to confidence (0.0-1.0)
    # Using a sigmoid-like curve for smoother transition
    max_score = 7.0
    if is_stego:
        confidence = 0.5 + (suspicion_score - 4) * 0.1  
    else:
        if suspicion_score <= 1:
            confidence = 0.9
        elif suspicion_score <= 2:
            confidence = 0.75
        else:
            confidence = 0.6
    
    # Ensure confidence is between 0.1 and 0.95
    confidence = max(0.1, min(0.95, confidence))
    
    # Add randomized noise to avoid constant values
    confidence_noise = random.uniform(-0.02, 0.02)
    confidence = max(0.1, min(0.95, confidence + confidence_noise))
    
    details['summary'] = {
        'suspicion_score': suspicion_score,
        'max_score': max_score,
        'is_stego': is_stego,
        'confidence': confidence,
        'quick_mode': quick
    }
    
    logger.info(f"LSB Analysis result for {file_path}: {'STEGO' if is_stego else 'CLEAN'} with confidence {confidence:.3f}")
    
    return is_stego, confidence, details

def main():
    if len(sys.argv) < 2:
        print("Usage: python lsboptimised.py <image_path> [--quick]")
        return
    
    file_path = sys.argv[1]
    quick_mode = "--quick" in sys.argv
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return
    
    is_stego, confidence, details = detect_stego(file_path, quick_mode)
    
    print("\nLSB Steganalysis Results:")
    print(f"Image: {file_path}")
    print(f"Result: {'STEGO' if is_stego else 'CLEAN'}")
    print(f"Confidence: {confidence:.3f}")
    
    if 'summary' in details:
        print(f"Suspicion Score: {details['summary']['suspicion_score']}/{details['summary']['max_score']}")
    
    print("\nDetailed Analysis:")
    if 'entropy' in details:
        print("LSB Entropy:")
        for channel, value in details['entropy'].items():
            print(f"  {channel.capitalize()}: {value:.4f}")
    
    if 'chi_square' in details:
        print("\nChi-Square Test:")
        for channel, data in details['chi_square'].items():
            print(f"  {channel.capitalize()}: {data['value']:.4f} ({data['result']})")
    
    if 'runs_test' in details:
        print("\nRuns Test:")
        for channel, data in details['runs_test'].items():
            print(f"  {channel.capitalize()}: {data['value']:.4f} ({data['result']})")

if __name__ == "__main__":
    main()