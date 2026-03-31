import struct
import numpy as np
from collections import defaultdict

def read_bmp_pixels(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        if len(file_data) < 54:  # Minimum BMP size
            print(f"File too small: {len(file_data)} bytes")
            return None

        signature = file_data[0:2]
        if signature != b'BM':
            print(f"Invalid BMP signature: {signature}")
            return None
        
        header_file_size, = struct.unpack('<I', file_data[2:6])
        data_offset, = struct.unpack('<I', file_data[10:14])
        dib_header_size, = struct.unpack('<I', file_data[14:18])
        
        print(f"BMP Header - File size: {header_file_size}, Data offset: {data_offset}")
        print(f"DIB header size: {dib_header_size}")
        
        if dib_header_size < 40: 
            print(f"Unsupported DIB header size: {dib_header_size}")
            return None
        
        width, height, planes, bits_per_pixel, compression = struct.unpack(
            '<iihhi', file_data[18:34]
        )
        
        print(f"Image properties - Width: {width}, Height: {abs(height)}, BPP: {bits_per_pixel}, Compression: {compression}")
        
        top_down = height < 0
        height = abs(height)
        
        if bits_per_pixel not in [24, 32] or compression != 0:
            print(f"Unsupported format: {bits_per_pixel}bpp, compression: {compression}")
            return None
        
        bytes_per_pixel = bits_per_pixel // 8
        bytes_per_row = width * bytes_per_pixel
        padding = (4 - (bytes_per_row % 4)) % 4
        padded_row_size = bytes_per_row + padding
        
        print(f"Row info - Bytes per pixel: {bytes_per_pixel}, Row size: {bytes_per_row}, Padding: {padding}")
        
        expected_data_size = height * padded_row_size
        if data_offset + expected_data_size > len(file_data):
            print(f"Insufficient data. Expected: {expected_data_size}, Available: {len(file_data) - data_offset}")
            return None
        
        pixel_data_raw = file_data[data_offset:data_offset + expected_data_size]
        
        pixels = []
        
        if bits_per_pixel == 24:
            # 24-bit BGR format
            for y in range(height):
                row_start = y * padded_row_size
                row_end = row_start + bytes_per_row
                row_data = pixel_data_raw[row_start:row_end]
                
                for x in range(0, len(row_data), 3):
                    if x + 2 < len(row_data):
                        b, g, r = row_data[x], row_data[x+1], row_data[x+2]
                        pixels.append((r, g, b))
        
        elif bits_per_pixel == 32:
            # Process 32-bit BGRA format
            for y in range(height):
                row_start = y * padded_row_size
                row_end = row_start + bytes_per_row
                row_data = pixel_data_raw[row_start:row_end]
                
                # Extract pixels from row in chunks of 4 bytes
                for x in range(0, len(row_data), 4):
                    if x + 3 < len(row_data):
                        b, g, r, a = row_data[x], row_data[x+1], row_data[x+2], row_data[x+3]
                        pixels.append((r, g, b)) 
        
        print(f"Successfully read {len(pixels)} pixels")
        
        return {
            'width': width,
            'height': height,
            'pixels': pixels,
            'total_pixels': len(pixels),
            'bits_per_pixel': bits_per_pixel,
            'top_down': top_down
        }
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except MemoryError:
        print(f"Not enough memory to load file: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading BMP file: {e}")
        return None

def extract_lsb_plane(pixel_data):
    if not pixel_data or 'pixels' not in pixel_data:
        return None
    
    pixels = pixel_data['pixels']
    num_pixels = len(pixels)
    
    red_lsbs = [0] * num_pixels
    green_lsbs = [0] * num_pixels
    blue_lsbs = [0] * num_pixels
    combined_lsbs = [0] * (num_pixels * 3)

    for i, (r, g, b) in enumerate(pixels):
        r_lsb = r & 1
        g_lsb = g & 1
        b_lsb = b & 1
        
        red_lsbs[i] = r_lsb
        green_lsbs[i] = g_lsb
        blue_lsbs[i] = b_lsb
        
        base_idx = i * 3
        combined_lsbs[base_idx] = r_lsb
        combined_lsbs[base_idx + 1] = g_lsb
        combined_lsbs[base_idx + 2] = b_lsb
    
    return {
        'red_lsbs': red_lsbs,
        'green_lsbs': green_lsbs,
        'blue_lsbs': blue_lsbs,
        'combined_lsbs': combined_lsbs
    }

def analyze_bit_distribution(lsb_plane):
    if not lsb_plane:
        return None
    
    analysis = {}
    
    for channel in ['red_lsbs', 'green_lsbs', 'blue_lsbs', 'combined_lsbs']:
        if channel not in lsb_plane:
            continue
        
        bits = lsb_plane[channel]
        total_bits = len(bits)
        
        if total_bits == 0:
            continue
        
        count_1 = sum(bits)
        count_0 = total_bits - count_1

        ratio_0 = count_0 / total_bits
        ratio_1 = count_1 / total_bits
        
        expected = total_bits * 0.5
        chi_square = ((count_0 - expected) ** 2 + (count_1 - expected) ** 2) / expected
        
        analysis[channel] = {
            'total_bits': total_bits,
            'count_0': count_0,
            'count_1': count_1,
            'ratio_0': ratio_0,
            'ratio_1': ratio_1,
            'chi_square': chi_square,
            'balance_score': abs(ratio_0 - 0.5)
        }
    
    return analysis

def calculate_entropy(bit_sequence):
    if not bit_sequence:
        return 0
    
    total = len(bit_sequence)
    count_1 = sum(bit_sequence)
    count_0 = total - count_1
    
    if count_0 == 0 or count_1 == 0:
        return 0
    
    p0 = count_0 / total
    p1 = count_1 / total
    
    import math
    try:
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    except (ValueError, ZeroDivisionError):
        entropy = 0
    
    return entropy

def log2_approx(x):
    if x <= 0:
        return 0
    if x == 1:
        return 0
    
    if x >= 1:
        int_part = x.bit_length() - 1
        frac_part = x / (1 << int_part) - 1
        return int_part + frac_part * 1.442695  # 1/ln(2)
    else:
        return -log2_approx(1/x)

def analyze_sequential_patterns(bit_sequence):
    if not bit_sequence or len(bit_sequence) < 2:
        return {}
    
    transitions_01 = 0
    transitions_10 = 0
    runs_0 = []
    runs_1 = []
    
    current_bit = bit_sequence[0]
    current_run_length = 1
    
    for i in range(1, len(bit_sequence)):
        bit = bit_sequence[i]
        if bit == current_bit:
            current_run_length += 1
        else:
            if current_bit == 0:
                runs_0.append(current_run_length)
                transitions_01 += 1
            else:
                runs_1.append(current_run_length)
                transitions_10 += 1
            
            current_bit = bit
            current_run_length = 1
    
    if current_bit == 0:
        runs_0.append(current_run_length)
    else:
        runs_1.append(current_run_length)
    
    total_transitions = transitions_01 + transitions_10
    sequence_length = len(bit_sequence)
    
    return {
        'transitions_01': transitions_01,
        'transitions_10': transitions_10,
        'total_transitions': total_transitions,
        'transition_rate': total_transitions / (sequence_length - 1) if sequence_length > 1 else 0,
        'avg_run_0': sum(runs_0) / len(runs_0) if runs_0 else 0,
        'avg_run_1': sum(runs_1) / len(runs_1) if runs_1 else 0,
        'max_run_0': max(runs_0, default=0),
        'max_run_1': max(runs_1, default=0),
        'runs_0_count': len(runs_0),
        'runs_1_count': len(runs_1)
    }

def analyze_block_patterns(bit_sequence, block_size=8):
    if not bit_sequence or len(bit_sequence) < block_size:
        return {}
    
    block_patterns = defaultdict(int)
    total_blocks = len(bit_sequence) // block_size
    
    for i in range(0, total_blocks * block_size, block_size):
        block = tuple(bit_sequence[i:i + block_size])
        block_patterns[block] += 1
    
    unique_block_count = len(block_patterns)
    
    block_entropy = 0
    if total_blocks > 0:
        inv_total = 1.0 / total_blocks
        for count in block_patterns.values():
            p = count * inv_total
            if p > 0:
                block_entropy -= p * log2_approx(p)
    
    most_common = sorted(block_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_blocks': total_blocks,
        'unique_blocks': unique_block_count,
        'block_entropy': block_entropy,
        'most_common_patterns': most_common,
        'pattern_diversity': unique_block_count / total_blocks if total_blocks > 0 else 0
    }

def calculate_autocorrelation(bit_sequence, max_lag=50):
    if not bit_sequence or len(bit_sequence) < 2:
        return []
    
    n = len(bit_sequence)
    max_lag = min(max_lag, n // 8)
    
    signal = [1 if bit else -1 for bit in bit_sequence]
    
    autocorr = [1.0]
    
    for lag in range(1, max_lag):
        valid_points = n - lag
        if valid_points <= 0:
            break
        
        correlation = sum(signal[i] * signal[i + lag] for i in range(valid_points)) / valid_points
        autocorr.append(correlation)
    
    return autocorr

def detect_periodic_patterns(bit_sequence):
    if not bit_sequence or len(bit_sequence) < 16:
        return {}
    
    n = len(bit_sequence)
    max_period = min(n // 8, 128)  
    periodic_scores = {}
    
    for period in range(2, max_period + 1):
        matches = sum(1 for i in range(period, n) if bit_sequence[i] == bit_sequence[i % period])
        comparisons = n - period
        
        if comparisons > 0:
            score = matches / comparisons
            if score > 0.5:
                periodic_scores[period] = score
    
    strong_periods = [(period, score) for period, score in periodic_scores.items() if score > 0.7]
    strong_periods.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'periodic_scores': periodic_scores,
        'strong_periods': strong_periods[:5],
        'max_periodicity': max(periodic_scores.values()) if periodic_scores else 0,
        'avg_periodicity': sum(periodic_scores.values()) / len(periodic_scores) if periodic_scores else 0
    }

def analyze_spatial_correlation(pixel_data):
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    width = pixel_data['width']
    height = pixel_data['height']
    pixels = pixel_data['pixels']
    
    if width < 2 or height < 2:
        return {}
    
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        lsb_grid = []
        for y in range(height):
            row = []
            for x in range(width):
                pixel_idx = y * width + x
                if pixel_idx < len(pixels):
                    pixel_value = pixels[pixel_idx][channel_idx]
                    row.append(pixel_value & 1)
            lsb_grid.append(row)
        
        # Calculate correlations
        horizontal_corr = calculate_2d_correlation(lsb_grid, direction='horizontal')
        vertical_corr = calculate_2d_correlation(lsb_grid, direction='vertical')
        
        results[channel_name] = {
            'horizontal_correlation': horizontal_corr,
            'vertical_correlation': vertical_corr,
            'diagonal_correlation': 0  
        }
    
    return results

def calculate_2d_correlation(grid, direction='horizontal'):
    if not grid or len(grid) < 2:
        return 0
    
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    if width < 2:
        return 0
    
    correlations = []
    
    if direction == 'horizontal':
        for y in range(height):
            row = grid[y]
            for x in range(width - 1):
                if row[x] == row[x + 1]:
                    correlations.append(1)
                else:
                    correlations.append(-1)
    
    elif direction == 'vertical':
        for y in range(height - 1):
            for x in range(width):
                if grid[y][x] == grid[y + 1][x]:
                    correlations.append(1)
                else:
                    correlations.append(-1)

    return sum(correlations) / len(correlations) if correlations else 0

def perform_runs_test(bit_sequence):
    if not bit_sequence or len(bit_sequence) < 2:
        return {}
    
    n = len(bit_sequence)
    n1 = sum(bit_sequence)  
    n0 = n - n1           
    
    if n1 == 0 or n0 == 0:
        return {'z_score': 0, 'p_value': 0, 'is_random': False}
    
    runs = 1
    for i in range(1, n):
        if bit_sequence[i] != bit_sequence[i-1]:
            runs += 1
    
    expected_runs = (2 * n1 * n0) / n + 1
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    
    if variance <= 0:
        return {'z_score': 0, 'p_value': 0, 'is_random': False}
    
    z_score = (runs - expected_runs) / (variance ** 0.5)
    
    is_random = abs(z_score) < 1.96  
    
    return {
        'runs': runs,
        'expected_runs': expected_runs,
        'z_score': z_score,
        'is_random': is_random,
        'variance': variance
    }

def analyze_frequency_domain(bit_sequence):
    if not bit_sequence or len(bit_sequence) < 8:
        return {}
    
    n = len(bit_sequence)
    
    transitions = sum(1 for i in range(1, n) if bit_sequence[i] != bit_sequence[i-1])
    frequency_score = transitions / (n - 1) if n > 1 else 0

    autocorr = calculate_autocorrelation(bit_sequence, max_lag=min(32, n//4))
    spectral_energy = sum(abs(ac) for ac in autocorr[1:]) if len(autocorr) > 1 else 0
    
    return {
        'transition_frequency': frequency_score,
        'spectral_energy': spectral_energy,
        'frequency_complexity': min(frequency_score * 2, 1.0)
    }

def cos_approx(x):
    x = x % (2 * 3.14159)  
    if x > 3.14159:
        x = 2 * 3.14159 - x
        sign = -1
    else:
        sign = 1

    x2 = x * x
    return sign * (1 - x2/2 + x2*x2/24 - x2*x2*x2/720)

def sin_approx(x):
    return cos_approx(x - 1.5708) 

def calculate_complexity_metrics(bit_sequence):
    if not bit_sequence:
        return {}
    
    n = len(bit_sequence)
    
    complexity = 1
    i = 0
    while i < n:
        j = i + 1
        while j <= n:
            substring = bit_sequence[i:j]
            if substring not in [bit_sequence[k:k+len(substring)] for k in range(i)]:
                complexity += 1
                i = j - 1
                break
            j += 1
        else:
            break
        i += 1
    
    max_complexity = n / log2_approx(n) if n > 1 else 1
    normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0
    
    return {
        'lempel_ziv_complexity': complexity,
        'normalized_complexity': min(normalized_complexity, 1.0),
        'sequence_length': n
    }

def detect_stego(file_path):
    print(f"Starting LSB steganography detection for: {file_path}")
    
    pixel_data = read_bmp_pixels(file_path)
    if not pixel_data:
        return {
            'is_stego': False,
            'confidence': 0,
            'error': 'Failed to read BMP file',
            'classification': 'error'
        }
    
    lsb_plane = extract_lsb_plane(pixel_data)
    if not lsb_plane:
        return {
            'is_stego': False,
            'confidence': 0,
            'error': 'Failed to extract LSB plane',
            'classification': 'error'
        }
    try:
        # Core analyses for detection
        bit_dist = analyze_bit_distribution(lsb_plane)
        entropy = calculate_entropy(lsb_plane['combined_lsbs'])
        sequential = analyze_sequential_patterns(lsb_plane['combined_lsbs'])
        runs_test = perform_runs_test(lsb_plane['combined_lsbs'])
        
        spatial = analyze_spatial_correlation(pixel_data)
        
        suspicion_factors = []
        
        # Chi-square test 
        if bit_dist and 'combined_lsbs' in bit_dist:
            chi_square = bit_dist['combined_lsbs']['chi_square']
            if chi_square < 1.0: 
                suspicion_factors.append(0.3)
        
        if entropy < 0.7 or entropy > 0.99:
            suspicion_factors.append(0.2)
        
        if runs_test and not runs_test.get('is_random', True):
            suspicion_factors.append(0.25)
        
        if spatial:
            avg_correlation = sum(
                abs(channel_data.get('horizontal_correlation', 0)) + 
                abs(channel_data.get('vertical_correlation', 0))
                for channel_data in spatial.values()
            ) / (len(spatial) * 2)
            
            if avg_correlation < 0.1: 
                suspicion_factors.append(0.25)
        
        total_suspicion = sum(suspicion_factors)
        confidence = min(total_suspicion, 1.0)
        
        if confidence > 0.6:
            classification = 'stego'
            is_stego = True
        elif confidence > 0.4:
            classification = 'suspicious'
            is_stego = True
        else:
            classification = 'clean'
            is_stego = False
        
        return {
            'is_stego': is_stego,
            'confidence': confidence,
            'classification': classification,
            'entropy': entropy,
            'bit_distribution': bit_dist,
            'sequential_patterns': sequential,
            'runs_test': runs_test,
            'spatial_correlation': spatial,
            'method': 'LSB'
        }
        
    except Exception as e:
        return {
            'is_stego': False,
            'confidence': 0,
            'error': f'Analysis failed: {str(e)}',
            'classification': 'error'
        }

def quick_classify(file_path):
    """Quick LSB classification for batch processing."""
    pixel_data = read_bmp_pixels(file_path)
    if not pixel_data:
        return 'error', 0
    
    lsb_plane = extract_lsb_plane(pixel_data)
    if not lsb_plane:
        return 'error', 0
    
    try:
        bit_dist = analyze_bit_distribution(lsb_plane)
        entropy = calculate_entropy(lsb_plane['combined_lsbs'])
        score = 0
        if bit_dist and 'combined_lsbs' in bit_dist:
            chi_square = bit_dist['combined_lsbs']['chi_square']
            if chi_square < 1.0:
                score += 0.4
        
        if entropy < 0.7 or entropy > 0.99:
            score += 0.3
        
        confidence = min(score, 1.0)
        classification = 'stego' if confidence > 0.5 else 'clean'
        
        return classification, confidence
        
    except Exception:
        return 'error', 0

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python lsb.py <image_path>")
        return
    
    image_path = sys.argv[1]
    result = detect_stego(image_path)
    
    print("\n=== LSB Analysis Results ===")
    print(f"Classification: {result.get('classification', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Is Stego: {result.get('is_stego', False)}")
    
    if 'entropy' in result:
        print(f"Entropy: {result['entropy']:.4f}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()