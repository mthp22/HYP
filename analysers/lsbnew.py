import struct
import math
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
        'combined_lsbs': combined_lsbs
    }

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
    
    try:
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    except (ValueError, ZeroDivisionError):
        entropy = 0
    
    return entropy

def chi_square_test(bit_sequence):
    """Perform chi-square test for randomness"""
    if not bit_sequence:
        return 0, 1.0
    
    total = len(bit_sequence)
    count_1 = sum(bit_sequence)
    count_0 = total - count_1
    
    expected = total / 2.0
    chi_square = ((count_0 - expected) ** 2 + (count_1 - expected) ** 2) / expected
    
    # Simple p-value approximation
    if chi_square < 0.004:  # Very low chi-square indicates too uniform (suspicious)
        p_value = 0.001
    elif chi_square < 3.841:  # 95% confidence threshold
        p_value = 0.1
    else:
        p_value = 0.001
    
    return chi_square, p_value

def runs_test(bit_sequence):
    """Wald-Wolfowitz runs test"""
    if not bit_sequence or len(bit_sequence) < 2:
        return 0, 1.0
    
    n = len(bit_sequence)
    n1 = sum(bit_sequence)
    n0 = n - n1
    
    if n1 == 0 or n0 == 0:
        return 0, 1.0
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if bit_sequence[i] != bit_sequence[i-1]:
            runs += 1
    
    # Expected runs and variance
    expected_runs = (2 * n1 * n0) / n + 1
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    
    if variance <= 0:
        return 0, 1.0
    
    # Z-score
    z_score = abs(runs - expected_runs) / math.sqrt(variance)
    
    # Convert to p-value (simplified)
    if z_score > 1.96:  # 95% confidence
        p_value = 0.05
    else:
        p_value = 0.5
    
    return z_score, p_value

def analyze_pairs_of_values(pixel_data):
    """Pairs of Values (PoV) attack for LSB steganography detection"""
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        # Count pairs and singles
        even_count = 0
        odd_count = 0
        
        for pixel in pixels:
            value = pixel[channel_idx]
            if value % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        
        total_pixels = len(pixels)
        
        # In natural images, there should be correlation between adjacent pixel values
        # In stego images with LSB replacement, this correlation is disrupted
        
        # Calculate expected vs actual distribution
        expected_even = total_pixels / 2
        expected_odd = total_pixels / 2
        
        chi_square_pov = ((even_count - expected_even) ** 2 / expected_even + 
                         (odd_count - expected_odd) ** 2 / expected_odd)
        
        # Calculate ratio
        if odd_count > 0:
            even_odd_ratio = even_count / odd_count
        else:
            even_odd_ratio = float('inf')
        
        results[channel_name] = {
            'even_count': even_count,
            'odd_count': odd_count,
            'even_odd_ratio': even_odd_ratio,
            'chi_square_pov': chi_square_pov,
            'balance_deviation': abs(0.5 - (even_count / total_pixels))
        }
    
    return results

def analyze_sample_pairs(pixel_data):
    """Sample Pairs Analysis for detecting LSB embedding"""
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        values = [pixel[channel_idx] for pixel in pixels]
        
        if len(values) < 2:
            continue
            
        # Count different types of pairs
        x_pairs = 0  # Pairs where values differ by 1 and LSBs are different
        y_pairs = 0  # Pairs where values are equal or differ by 1 and LSBs are same
        z_pairs = 0  # Other pairs
        
        for i in range(0, len(values) - 1, 2):
            v1, v2 = values[i], values[i + 1]
            
            # Check LSBs
            lsb1, lsb2 = v1 & 1, v2 & 1
            
            # Check if values form a "close pair" (differ by at most 1)
            diff = abs(v1 - v2)
            
            if diff <= 1:
                if lsb1 != lsb2:
                    x_pairs += 1
                else:
                    y_pairs += 1
            else:
                z_pairs += 1
        
        total_pairs = x_pairs + y_pairs + z_pairs
        
        if total_pairs > 0:
            # In natural images, x_pairs should be roughly equal to y_pairs
            # In stego images, this balance is disrupted
            balance_ratio = x_pairs / y_pairs if y_pairs > 0 else float('inf')
            
            results[channel_name] = {
                'x_pairs': x_pairs,
                'y_pairs': y_pairs,
                'z_pairs': z_pairs,
                'total_pairs': total_pairs,
                'balance_ratio': balance_ratio,
                'x_ratio': x_pairs / total_pairs,
                'y_ratio': y_pairs / total_pairs,
                'imbalance_score': abs(x_pairs - y_pairs) / total_pairs
            }
    
    return results

def analyze_histogram_center_of_mass(pixel_data):
    """Analyze center of mass of histograms"""
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        # Create histogram
        histogram = [0] * 256
        for pixel in pixels:
            value = pixel[channel_idx]
            histogram[value] += 1
        
        # Calculate center of mass
        total_pixels = sum(histogram)
        if total_pixels == 0:
            continue
            
        center_of_mass = sum(i * count for i, count in enumerate(histogram)) / total_pixels
        
        # Calculate variance around center of mass
        variance = sum(((i - center_of_mass) ** 2) * count for i, count in enumerate(histogram)) / total_pixels
        
        # Analyze even/odd distribution
        even_sum = sum(histogram[i] for i in range(0, 256, 2))
        odd_sum = sum(histogram[i] for i in range(1, 256, 2))
        
        results[channel_name] = {
            'center_of_mass': center_of_mass,
            'variance': variance,
            'even_sum': even_sum,
            'odd_sum': odd_sum,
            'even_odd_ratio': even_sum / odd_sum if odd_sum > 0 else float('inf'),
            'even_odd_diff': abs(even_sum - odd_sum)
        }
    
    return results

def analyze_bit_plane_complexity(pixel_data):
    """Analyze complexity of bit planes"""
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    pixels = pixel_data['pixels']
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        bit_planes = [[] for _ in range(8)]
        
        # Extract bit planes
        for pixel in pixels:
            value = pixel[channel_idx]
            for bit_pos in range(8):
                bit_planes[bit_pos].append((value >> bit_pos) & 1)
        
        plane_complexities = []
        
        for bit_pos, plane in enumerate(bit_planes):
            if not plane:
                continue
                
            # Count transitions (changes between adjacent bits)
            transitions = sum(1 for i in range(1, len(plane)) if plane[i] != plane[i-1])
            complexity = transitions / (len(plane) - 1) if len(plane) > 1 else 0
            plane_complexities.append(complexity)
        
        if plane_complexities:
            lsb_complexity = plane_complexities[0]  # LSB is bit position 0
            avg_other_complexity = sum(plane_complexities[1:]) / len(plane_complexities[1:]) if len(plane_complexities) > 1 else 0
            
            results[channel_name] = {
                'lsb_complexity': lsb_complexity,
                'avg_other_complexity': avg_other_complexity,
                'complexity_ratio': lsb_complexity / avg_other_complexity if avg_other_complexity > 0 else 0,
                'complexity_difference': abs(lsb_complexity - avg_other_complexity)
            }
    
    return results

def detect_stego(file_path):
    print(f"Starting advanced LSB steganography detection for: {file_path}")
    
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
        # Perform multiple analyses
        print("Performing chi-square test...")
        chi_square, chi_p_value = chi_square_test(lsb_plane['combined_lsbs'])
        
        print("Performing runs test...")
        runs_z, runs_p_value = runs_test(lsb_plane['combined_lsbs'])
        
        print("Calculating entropy...")
        entropy = calculate_entropy(lsb_plane['combined_lsbs'])
        
        print("Analyzing pairs of values...")
        pov_analysis = analyze_pairs_of_values(pixel_data)
        
        print("Analyzing sample pairs...")
        sp_analysis = analyze_sample_pairs(pixel_data)
        
        print("Analyzing histogram center of mass...")
        histogram_analysis = analyze_histogram_center_of_mass(pixel_data)
        
        print("Analyzing bit plane complexity...")
        complexity_analysis = analyze_bit_plane_complexity(pixel_data)
        
        # Scoring system
        suspicion_score = 0
        max_score = 7
        
        # Chi-square test (low values indicate too uniform distribution)
        if chi_square < 10.0:  # Threshold for suspicion
            suspicion_score += (10.0 - chi_square) / 10.0
            print(f"Chi-square suspicious: {chi_square:.3f}")
        
        # Runs test (low p-value indicates non-randomness)
        if runs_p_value < 0.1:
            suspicion_score += 1.0
            print(f"Runs test suspicious: p={runs_p_value:.3f}")
        
        # Entropy (too high or too low)
        if entropy > 0.99 or entropy < 0.95:
            suspicion_score += 1.0
            print(f"Entropy suspicious: {entropy:.3f}")
        
        # Pairs of Values analysis
        if pov_analysis:
            avg_balance_deviation = sum(data['balance_deviation'] for data in pov_analysis.values()) / len(pov_analysis)
            if avg_balance_deviation < 0.05:  # Too balanced
                suspicion_score += 1.0
                print(f"PoV analysis suspicious: deviation={avg_balance_deviation:.3f}")
        
        # Sample Pairs analysis
        if sp_analysis:
            avg_imbalance = sum(data['imbalance_score'] for data in sp_analysis.values()) / len(sp_analysis)
            if avg_imbalance > 0.1:  # Significant imbalance
                suspicion_score += 1.0
                print(f"Sample pairs suspicious: imbalance={avg_imbalance:.3f}")
        
        # Histogram analysis
        if histogram_analysis:
            avg_even_odd_diff = sum(data['even_odd_diff'] for data in histogram_analysis.values()) / len(histogram_analysis)
            total_pixels = pixel_data['total_pixels']
            relative_diff = avg_even_odd_diff / total_pixels
            if relative_diff < 0.02:  # Too balanced
                suspicion_score += 1.0
                print(f"Histogram suspicious: relative_diff={relative_diff:.3f}")
        
        # Bit plane complexity
        if complexity_analysis:
            avg_complexity_ratio = sum(data['complexity_ratio'] for data in complexity_analysis.values() if data['complexity_ratio'] > 0) / len([d for d in complexity_analysis.values() if d['complexity_ratio'] > 0])
            if avg_complexity_ratio > 1.2:  # LSB more complex than other planes
                suspicion_score += 1.0
                print(f"Complexity suspicious: ratio={avg_complexity_ratio:.3f}")
        
        confidence = suspicion_score / max_score
        
        # Classification
        if confidence > 0.6:
            classification = 'stego'
            is_stego = True
        elif confidence > 0.4:
            classification = 'suspicious'
            is_stego = True
        else:
            classification = 'clean'
            is_stego = False
        
        print(f"Analysis complete - Suspicion score: {suspicion_score:.3f}/{max_score}")
        
        return {
            'is_stego': is_stego,
            'confidence': confidence,
            'classification': classification,
            'suspicion_score': suspicion_score,
            'max_score': max_score,
            'entropy': entropy,
            'chi_square': chi_square,
            'chi_p_value': chi_p_value,
            'runs_z_score': runs_z,
            'runs_p_value': runs_p_value,
            'pov_analysis': pov_analysis,
            'sample_pairs': sp_analysis,
            'histogram_analysis': histogram_analysis,
            'complexity_analysis': complexity_analysis,
            'method': 'Advanced LSB Detection'
        }
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return {
            'is_stego': False,
            'confidence': 0,
            'error': f'Analysis failed: {str(e)}',
            'classification': 'error'
        }

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python lsbnew.py <image_path>")
        return
    
    image_path = sys.argv[1]
    result = detect_stego(image_path)
    
    print("\n=== Advanced LSB Analysis Results ===")
    print(f"Classification: {result.get('classification', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Is Stego: {result.get('is_stego', False)}")
    
    if 'entropy' in result:
        print(f"Entropy: {result['entropy']:.4f}")
    if 'chi_square' in result:
        print(f"Chi-square: {result['chi_square']:.4f}")
    if 'suspicion_score' in result:
        print(f"Suspicion Score: {result['suspicion_score']:.3f}/{result.get('max_score', 0)}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()