import struct
import math
import sys
from collections import defaultdict

def read_bmp_pixels(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        if len(file_data) < 54:
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
            for y in range(height):
                row_start = y * padded_row_size
                row_end = row_start + bytes_per_row
                row_data = pixel_data_raw[row_start:row_end]
                
                for x in range(0, len(row_data), 3):
                    if x + 2 < len(row_data):
                        b, g, r = row_data[x], row_data[x+1], row_data[x+2]
                        pixels.append((r, g, b))
        
        elif bits_per_pixel == 32:
            for y in range(height):
                row_start = y * padded_row_size
                row_end = row_start + bytes_per_row
                row_data = pixel_data_raw[row_start:row_end]
                
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
    
    # Pre-allocate arrays for LSBs
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
    
    try:
        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    except (ValueError, ZeroDivisionError):
        entropy = 0
    
    return entropy

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

def analyze_local_pixel_neighborhoods(pixel_data, neighborhood_size=3, sample_rate=0.1):
    #Analyze LSB patterns in local pixel neighborhoods 
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    width = pixel_data['width']
    height = pixel_data['height']
    pixels = pixel_data['pixels']
    
    if width < neighborhood_size or height < neighborhood_size:
        return {}
    
    channel_correlations = {'red': [], 'green': [], 'blue': []}
    
    # Sample neighborhoods 
    max_neighborhoods = int((height - neighborhood_size + 1) * (width - neighborhood_size + 1) * sample_rate)
    step_size = max(1, int(1 / sample_rate))
    
    analyzed = 0
    for y in range(0, height - neighborhood_size + 1, step_size):
        for x in range(0, width - neighborhood_size + 1, step_size):
            if analyzed >= max_neighborhoods:
                break
                
            neighborhood = []
            for dy in range(neighborhood_size):
                for dx in range(neighborhood_size):
                    pixel_idx = (y + dy) * width + (x + dx)
                    if pixel_idx < len(pixels):
                        neighborhood.append(pixels[pixel_idx])
            
            if len(neighborhood) == neighborhood_size * neighborhood_size:
                for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
                    lsbs = [pixel[channel_idx] & 1 for pixel in neighborhood]
                    
                    count_1 = sum(lsbs)
                    count_0 = len(lsbs) - count_1
                    
                    if count_0 > 0 and count_1 > 0:
                        p0, p1 = count_0 / len(lsbs), count_1 / len(lsbs)
                        local_entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
                    else:
                        local_entropy = 0
                    
                    center_idx = len(lsbs) // 2
                    center_lsb = lsbs[center_idx]
                    correlations = [1 if lsb == center_lsb else 0 for i, lsb in enumerate(lsbs) if i != center_idx]
                    local_correlation = sum(correlations) / len(correlations) if correlations else 0
                    
                    channel_correlations[channel_name].append({
                        'entropy': local_entropy,
                        'correlation': local_correlation
                    })
            
            analyzed += 1
        
        if analyzed >= max_neighborhoods:
            break
    
    # Aggregate results
    results = {}
    for channel_name in ['red', 'green', 'blue']:
        correlations = channel_correlations[channel_name]
        if correlations:
            entropies = [c['entropy'] for c in correlations]
            corrs = [c['correlation'] for c in correlations]
            
            avg_entropy = sum(entropies) / len(entropies)
            results[channel_name] = {
                'avg_local_entropy': avg_entropy,
                'std_local_entropy': math.sqrt(sum((e - avg_entropy)**2 for e in entropies) / len(entropies)),
                'avg_local_correlation': sum(corrs) / len(corrs),
                'low_entropy_neighborhoods': sum(1 for e in entropies if e < 0.5) / len(entropies)
            }
    
    return results

def analyze_bit_plane_noise(pixel_data, sample_rate=0.2):  
    #Analyze noise characteristics across different bit planes
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    pixels = pixel_data['pixels']
    if not pixels:
        return {}
    
    sample_size = min(len(pixels), int(len(pixels) * sample_rate))
    step = max(1, len(pixels) // sample_size)
    sampled_pixels = pixels[::step]
    
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        bit_planes = [[] for _ in range(8)]
        
        for pixel in sampled_pixels:
            value = pixel[channel_idx]
            for bit_pos in range(8):
                bit_planes[bit_pos].append((value >> bit_pos) & 1)
        
        plane_analysis = {}
        for bit_pos in range(8):
            bits = bit_planes[bit_pos]
            
            count_1 = sum(bits)
            count_0 = len(bits) - count_1
            
            if count_0 > 0 and count_1 > 0:
                p0, p1 = count_0 / len(bits), count_1 / len(bits)
                entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
            else:
                entropy = 0
            
            transitions = sum(1 for i in range(1, len(bits)) if bits[i] != bits[i-1])
            transition_rate = transitions / (len(bits) - 1) if len(bits) > 1 else 0
            
            plane_analysis[f'bit_{bit_pos}'] = {
                'entropy': entropy,
                'transition_rate': transition_rate,
                'balance': abs(0.5 - count_1 / len(bits))
            }
        
        lsb_entropy = plane_analysis['bit_0']['entropy']
        higher_bit_avg_entropy = sum(plane_analysis[f'bit_{i}']['entropy'] for i in range(1, 4)) / 3
        
        results[channel_name] = {
            'bit_planes': plane_analysis,
            'lsb_entropy': lsb_entropy,
            'higher_bit_avg_entropy': higher_bit_avg_entropy,
            'entropy_ratio': lsb_entropy / higher_bit_avg_entropy if higher_bit_avg_entropy > 0 else 0,
            'noise_inconsistency': abs(lsb_entropy - higher_bit_avg_entropy)
        }
    
    return results

def calculate_cooccurrence_matrix(bit_sequence, distance=1):
   #Calculate co-occurrence matrix for bit patterns
    if not bit_sequence or len(bit_sequence) < distance + 1:
        return {}
    
    if len(bit_sequence) > 10000:
        step = len(bit_sequence) // 10000
        bit_sequence = bit_sequence[::step]
    
    cooc_matrix = [[0, 0], [0, 0]]
    
    for i in range(len(bit_sequence) - distance):
        bit1 = bit_sequence[i]
        bit2 = bit_sequence[i + distance]
        cooc_matrix[bit1][bit2] += 1
    
    total = sum(sum(row) for row in cooc_matrix)
    if total == 0:
        return {}
    
    normalized_matrix = [[cooc_matrix[i][j] / total for j in range(2)] for i in range(2)]
    
    contrast = sum(abs(i - j) * normalized_matrix[i][j] for i in range(2) for j in range(2))
    energy = sum(normalized_matrix[i][j]**2 for i in range(2) for j in range(2))
    homogeneity = sum(normalized_matrix[i][j] / (1 + abs(i - j)) for i in range(2) for j in range(2))
    
    return {
        'matrix': normalized_matrix,
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity,
        'entropy': -sum(normalized_matrix[i][j] * math.log2(normalized_matrix[i][j]) 
                       for i in range(2) for j in range(2) 
                       if normalized_matrix[i][j] > 0)
    }

def analyze_spatial_correlation(pixel_data, sample_rate=0.3):  
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    width = pixel_data['width']
    height = pixel_data['height']
    pixels = pixel_data['pixels']
    
    if width < 2 or height < 2:
        return {}
    
    results = {}
    
    for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
        sample_height = max(2, int(height * sample_rate))
        sample_width = max(2, int(width * sample_rate))
        
        lsb_grid = []
        for y in range(0, height, height // sample_height):
            row = []
            for x in range(0, width, width // sample_width):
                pixel_idx = y * width + x
                if pixel_idx < len(pixels):
                    pixel_value = pixels[pixel_idx][channel_idx]
                    row.append(pixel_value & 1)
                else:
                    row.append(0)
            lsb_grid.append(row)
        
        horizontal_corr = calculate_2d_correlation(lsb_grid, direction='horizontal')
        vertical_corr = calculate_2d_correlation(lsb_grid, direction='vertical')
        diagonal_corr = calculate_2d_correlation(lsb_grid, direction='diagonal')
        
        flat_lsbs = [bit for row in lsb_grid for bit in row]
        cooc_horizontal = calculate_cooccurrence_matrix(flat_lsbs, distance=1)
        
        results[channel_name] = {
            'horizontal_correlation': horizontal_corr,
            'vertical_correlation': vertical_corr,
            'diagonal_correlation': diagonal_corr,
            'cooccurrence_horizontal': cooc_horizontal
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
    
    elif direction == 'diagonal':
        for y in range(height - 1):
            for x in range(width - 1):
                if grid[y][x] == grid[y + 1][x + 1]:
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

def calculate_complexity_metrics(bit_sequence):
    if not bit_sequence:
        return {}
    
    n = len(bit_sequence)
    

    if n > 5000: 
        step = n // 5000
        bit_sequence = bit_sequence[::step]
        n = len(bit_sequence)
    
    complexity = 1
    i = 0
    
    while i < n and complexity < 100:  
        j = i + 1
        found = False
        
        while j <= min(n, i + 50):  
            substring = bit_sequence[i:j]
            
            for k in range(max(0, i - 100), i):  
                if k + len(substring) <= i:
                    if bit_sequence[k:k+len(substring)] == substring:
                        found = True
                        break
            
            if not found:
                complexity += 1
                i = j - 1
                break
            j += 1
        else:
            break
        i += 1
    
    theoretical_max = n / math.log2(n) if n > 1 else 1
    normalized_complexity = complexity / theoretical_max if theoretical_max > 0 else 0
    
    return {
        'lempel_ziv_complexity': complexity,
        'normalized_complexity': min(normalized_complexity, 1.0),
        'sequence_length': n
    }

def detect_stego(file_path):
    print(f"Starting enhanced LSB steganography detection for: {file_path}")
    
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
        print("Analyzing bit distribution...")
        bit_dist = analyze_bit_distribution(lsb_plane)
        
        print("Calculating entropy...")
        entropy = calculate_entropy(lsb_plane['combined_lsbs'])
        
        print("Analyzing sequential patterns...")
        sequential = analyze_sequential_patterns(lsb_plane['combined_lsbs'])
        
        print("Performing runs test...")
        runs_test = perform_runs_test(lsb_plane['combined_lsbs'])
        
        print("Analyzing spatial correlation...")
        spatial = analyze_spatial_correlation(pixel_data)
        
        print("Analyzing local neighborhoods...")
        local_neighborhoods = analyze_local_pixel_neighborhoods(pixel_data)
        
        print("Analyzing bit plane noise...")
        bit_plane_noise = analyze_bit_plane_noise(pixel_data)
        
        print("Calculating complexity metrics...")
        complexity = calculate_complexity_metrics(lsb_plane['combined_lsbs'])
        
        suspicion_factors = []
        
        # Scoring system
        if bit_dist and 'combined_lsbs' in bit_dist:
            chi_square = bit_dist['combined_lsbs']['chi_square']
            if chi_square < 2.71:
                suspicion_factors.append(min(0.4, (2.71 - chi_square) / 2.71 * 0.4))
        
        if entropy < 0.8 or entropy > 0.98:
            if entropy < 0.8:
                suspicion_factors.append(0.3 * (0.8 - entropy) / 0.8)
            else:
                suspicion_factors.append(0.3 * (entropy - 0.98) / 0.02)
        
        if runs_test and not runs_test.get('is_random', True):
            z_score = abs(runs_test.get('z_score', 0))
            if z_score > 1.96:
                suspicion_factors.append(min(0.3, (z_score - 1.96) / 4.0 * 0.3))
        
        if spatial:
            avg_correlation = 0
            correlation_count = 0
            for channel_data in spatial.values():
                for corr_type in ['horizontal_correlation', 'vertical_correlation', 'diagonal_correlation']:
                    if corr_type in channel_data:
                        avg_correlation += abs(channel_data[corr_type])
                        correlation_count += 1
            
            if correlation_count > 0:
                avg_correlation /= correlation_count
                if avg_correlation < 0.05:
                    suspicion_factors.append(0.25)
        
        if local_neighborhoods:
            low_entropy_ratio = 0
            channel_count = 0
            for channel_data in local_neighborhoods.values():
                if 'low_entropy_neighborhoods' in channel_data:
                    low_entropy_ratio += channel_data['low_entropy_neighborhoods']
                    channel_count += 1
            
            if channel_count > 0:
                avg_low_entropy_ratio = low_entropy_ratio / channel_count
                if avg_low_entropy_ratio > 0.3:
                    suspicion_factors.append(0.2 * avg_low_entropy_ratio)
        
        if bit_plane_noise:
            noise_inconsistencies = []
            for channel_data in bit_plane_noise.values():
                if 'noise_inconsistency' in channel_data:
                    noise_inconsistencies.append(channel_data['noise_inconsistency'])
            
            if noise_inconsistencies:
                avg_inconsistency = sum(noise_inconsistencies) / len(noise_inconsistencies)
                if avg_inconsistency > 0.2:
                    suspicion_factors.append(min(0.25, avg_inconsistency * 1.25))
        
        if complexity:
            normalized_complexity = complexity.get('normalized_complexity', 0)
            if normalized_complexity < 0.3:
                suspicion_factors.append(0.2 * (0.3 - normalized_complexity) / 0.3)
        
        total_suspicion = sum(suspicion_factors)
        confidence = min(total_suspicion, 1.0)
        
        if confidence > 0.7:
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
            'local_neighborhoods': local_neighborhoods,
            'bit_plane_noise': bit_plane_noise,
            'complexity_metrics': complexity,
            'suspicion_factors': suspicion_factors,
            'method': 'Enhanced LSB'
        }
        
    except Exception as e:
        return {
            'is_stego': False,
            'confidence': 0,
            'error': f'Analysis failed: {str(e)}',
            'classification': 'error'
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: python lsboptimised.py <image_path>")
        return
    
    image_path = sys.argv[1]
    result = detect_stego(image_path)
    
    print("\n=== Enhanced LSB Analysis Results ===")
    print(f"Classification: {result.get('classification', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Is Stego: {result.get('is_stego', False)}")
    
    if 'entropy' in result:
        print(f"Entropy: {result['entropy']:.4f}")
    
    if 'suspicion_factors' in result:
        print(f"Suspicion factors: {[f'{f:.3f}' for f in result['suspicion_factors']]}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()