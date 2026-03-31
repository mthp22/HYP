def read_bmp_pixels(file_path):
    """Read raw pixel data from BMP file (uncompressed)."""
    try:
        with open(file_path, 'rb') as f:
            # Check file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to beginning
            
            if file_size < 54:  # Minimum BMP size
                print(f"File too small: {file_size} bytes")
                return None
            
            # Read BMP header (14 bytes)
            bmp_header = f.read(14)
            if len(bmp_header) < 14:
                print("Failed to read BMP header")
                return None
            
            # Check BMP signature
            signature = bmp_header[0:2]
            if signature != b'BM':
                print(f"Invalid BMP signature: {signature}")
                return None
            
            # Extract file size and data offset
            header_file_size = int.from_bytes(bmp_header[2:6], byteorder='little')
            data_offset = int.from_bytes(bmp_header[10:14], byteorder='little')
            
            print(f"BMP Header - File size: {header_file_size}, Data offset: {data_offset}")
            
            # Read DIB header size first
            dib_size_bytes = f.read(4)
            if len(dib_size_bytes) < 4:
                print("Failed to read DIB header size")
                return None
            
            dib_header_size = int.from_bytes(dib_size_bytes, byteorder='little')
            print(f"DIB header size: {dib_header_size}")
            
            # Read rest of DIB header
            remaining_dib = f.read(dib_header_size - 4)
            if len(remaining_dib) < dib_header_size - 4:
                print(f"Failed to read complete DIB header. Expected {dib_header_size - 4}, got {len(remaining_dib)}")
                return None
            
            dib_header = dib_size_bytes + remaining_dib
            
            # Extract image properties based on DIB header type
            if dib_header_size >= 40:  # BITMAPINFOHEADER or later
                width = int.from_bytes(dib_header[4:8], byteorder='little', signed=True)
                height = int.from_bytes(dib_header[8:12], byteorder='little', signed=True)
                planes = int.from_bytes(dib_header[12:14], byteorder='little')
                bits_per_pixel = int.from_bytes(dib_header[14:16], byteorder='little')
                compression = int.from_bytes(dib_header[16:20], byteorder='little')
                
                print(f"Image properties - Width: {width}, Height: {height}, BPP: {bits_per_pixel}, Compression: {compression}")
                
                # Handle negative height (top-down bitmap)
                if height < 0:
                    height = abs(height)
                    top_down = True
                else:
                    top_down = False
                
                # Support multiple bit depths
                if bits_per_pixel not in [24, 32]:
                    print(f"Unsupported bit depth: {bits_per_pixel}. Only 24-bit and 32-bit BMPs are supported.")
                    return None
                
                # Only support uncompressed BMPs
                if compression != 0:
                    print(f"Unsupported compression: {compression}. Only uncompressed BMPs are supported.")
                    return None
                
                bytes_per_pixel = bits_per_pixel // 8
                
                # Calculate row padding (BMP rows are padded to 4-byte boundaries)
                bytes_per_row = width * bytes_per_pixel
                padding = (4 - (bytes_per_row % 4)) % 4
                padded_row_size = bytes_per_row + padding
                
                print(f"Row info - Bytes per pixel: {bytes_per_pixel}, Row size: {bytes_per_row}, Padding: {padding}, Padded size: {padded_row_size}")
                
                # Seek to pixel data
                f.seek(data_offset)
                
                # Read pixel data
                pixel_data = []
                
                for y in range(height):
                    row_data = f.read(padded_row_size)
                    if len(row_data) < padded_row_size:
                        print(f"Failed to read complete row {y}. Expected {padded_row_size}, got {len(row_data)}")
                        return None
                    
                    # Process pixels in this row
                    for x in range(width):
                        pixel_offset = x * bytes_per_pixel
                        if pixel_offset + bytes_per_pixel - 1 < len(row_data):
                            if bits_per_pixel == 24:
                                # 24-bit BMP: BGR format
                                b = row_data[pixel_offset]
                                g = row_data[pixel_offset + 1]
                                r = row_data[pixel_offset + 2]
                            elif bits_per_pixel == 32:
                                # 32-bit BMP: BGRA format
                                b = row_data[pixel_offset]
                                g = row_data[pixel_offset + 1]
                                r = row_data[pixel_offset + 2]
                                # Ignore alpha channel for now
                            
                            pixel_data.append((r, g, b))
                
                print(f"Successfully read {len(pixel_data)} pixels")
                
                return {
                    'width': width,
                    'height': height,
                    'pixels': pixel_data,
                    'total_pixels': len(pixel_data),
                    'bits_per_pixel': bits_per_pixel,
                    'top_down': top_down
                }
            
            else:
                print(f"Unsupported DIB header size: {dib_header_size}")
                return None
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Permission denied: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading BMP file: {e}")
        return None

def extract_lsb_plane(pixel_data):
    """Extract the least significant bit plane from RGB pixels."""
    if not pixel_data or 'pixels' not in pixel_data:
        return None
    
    lsb_data = {
        'red_lsbs': [],
        'green_lsbs': [],
        'blue_lsbs': [],
        'combined_lsbs': []
    }
    
    for r, g, b in pixel_data['pixels']:
        # Extract LSB from each channel
        r_lsb = r & 1
        g_lsb = g & 1
        b_lsb = b & 1
        
        lsb_data['red_lsbs'].append(r_lsb)
        lsb_data['green_lsbs'].append(g_lsb)
        lsb_data['blue_lsbs'].append(b_lsb)
        lsb_data['combined_lsbs'].extend([r_lsb, g_lsb, b_lsb])
    
    return lsb_data

def analyze_bit_distribution(lsb_plane):
    """Analyze distribution of 0s and 1s in LSB layer."""
    if not lsb_plane:
        return None
    
    analysis = {}
    
    # Analyze each channel separately
    for channel in ['red_lsbs', 'green_lsbs', 'blue_lsbs', 'combined_lsbs']:
        if channel not in lsb_plane:
            continue
        
        bits = lsb_plane[channel]
        total_bits = len(bits)
        
        if total_bits == 0:
            continue
        
        # Count 0s and 1s
        count_0 = bits.count(0)
        count_1 = bits.count(1)
        
        # Calculate basic statistics
        ratio_0 = count_0 / total_bits
        ratio_1 = count_1 / total_bits
        
        # Calculate chi-square test for uniform distribution
        expected = total_bits / 2
        chi_square = ((count_0 - expected) ** 2 / expected) + ((count_1 - expected) ** 2 / expected)
        
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
    """Calculate Shannon entropy of bit sequence."""
    if not bit_sequence:
        return 0
    
    total = len(bit_sequence)
    count_0 = bit_sequence.count(0)
    count_1 = bit_sequence.count(1)
    
    if count_0 == 0 or count_1 == 0:
        return 0
    
    p0 = count_0 / total
    p1 = count_1 / total
    
    # More accurate entropy calculation
    if p0 > 0 and p1 > 0:
        # Approximate log2 using bit operations and lookup
        entropy = -(p0 * log2_approx(p0) + p1 * log2_approx(p1))
    else:
        entropy = 0
    
    return entropy

def log2_approx(x):
    """Approximate log2 calculation without imports."""
    if x <= 0:
        return 0
    
    # Use bit manipulation for rough log2 approximation
    if x >= 1:
        return x.bit_length() - 1
    else:
        # For fractional values, use series approximation
        # log2(x) ≈ (x-1)/ln(2) for x close to 1
        # ln(2) ≈ 0.693147
        return (x - 1) / 0.693147

def analyze_sequential_patterns(bit_sequence):
    """Analyze sequential patterns in LSB data."""
    if not bit_sequence or len(bit_sequence) < 2:
        return {}
    
    # Count transitions
    transitions_01 = 0
    transitions_10 = 0
    runs_0 = []
    runs_1 = []
    
    current_bit = bit_sequence[0]
    current_run_length = 1
    
    for i in range(1, len(bit_sequence)):
        if bit_sequence[i] == current_bit:
            current_run_length += 1
        else:
            # Record run length
            if current_bit == 0:
                runs_0.append(current_run_length)
                transitions_01 += 1
            else:
                runs_1.append(current_run_length)
                transitions_10 += 1
            
            current_bit = bit_sequence[i]
            current_run_length = 1
    
    # Record final run
    if current_bit == 0:
        runs_0.append(current_run_length)
    else:
        runs_1.append(current_run_length)
    
    # Calculate statistics
    total_transitions = transitions_01 + transitions_10
    avg_run_0 = sum(runs_0) / len(runs_0) if runs_0 else 0
    avg_run_1 = sum(runs_1) / len(runs_1) if runs_1 else 0
    max_run_0 = max(runs_0) if runs_0 else 0
    max_run_1 = max(runs_1) if runs_1 else 0
    
    return {
        'transitions_01': transitions_01,
        'transitions_10': transitions_10,
        'total_transitions': total_transitions,
        'transition_rate': total_transitions / (len(bit_sequence) - 1) if len(bit_sequence) > 1 else 0,
        'avg_run_0': avg_run_0,
        'avg_run_1': avg_run_1,
        'max_run_0': max_run_0,
        'max_run_1': max_run_1,
        'runs_0_count': len(runs_0),
        'runs_1_count': len(runs_1)
    }

def analyze_block_patterns(bit_sequence, block_size=8):
    """Analyze patterns in blocks of LSB data."""
    if not bit_sequence or len(bit_sequence) < block_size:
        return {}
    
    block_patterns = {}
    unique_blocks = set()
    
    # Extract blocks and count patterns
    for i in range(0, len(bit_sequence) - block_size + 1, block_size):
        block = tuple(bit_sequence[i:i + block_size])
        unique_blocks.add(block)
        
        if block in block_patterns:
            block_patterns[block] += 1
        else:
            block_patterns[block] = 1
    
    total_blocks = len(bit_sequence) // block_size
    unique_block_count = len(unique_blocks)
    
    # Calculate block entropy
    block_entropy = 0
    if total_blocks > 0:
        for count in block_patterns.values():
            if count > 0:
                p = count / total_blocks
                block_entropy -= p * log2_approx(p)
    
    # Find most common patterns
    sorted_patterns = sorted(block_patterns.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_patterns[:5] if len(sorted_patterns) >= 5 else sorted_patterns
    
    return {
        'total_blocks': total_blocks,
        'unique_blocks': unique_block_count,
        'block_entropy': block_entropy,
        'most_common_patterns': most_common,
        'pattern_diversity': unique_block_count / total_blocks if total_blocks > 0 else 0
    }

def calculate_autocorrelation(bit_sequence, max_lag=100):
    """Calculate autocorrelation of bit sequence."""
    if not bit_sequence or len(bit_sequence) < 2:
        return []
    
    n = len(bit_sequence)
    max_lag = min(max_lag, n // 4)  # Limit lag to prevent overfitting
    
    # Convert bits to -1, 1 for better correlation analysis
    signal = [1 if bit else -1 for bit in bit_sequence]
    
    autocorr = []
    for lag in range(max_lag):
        if lag == 0:
            # Zero lag is always 1
            autocorr.append(1.0)
        else:
            # Calculate correlation for this lag
            sum_product = 0
            valid_points = n - lag
            
            for i in range(valid_points):
                sum_product += signal[i] * signal[i + lag]
            
            correlation = sum_product / valid_points if valid_points > 0 else 0
            autocorr.append(correlation)
    
    return autocorr

def detect_periodic_patterns(bit_sequence):
    """Detect periodic patterns in LSB data."""
    if not bit_sequence or len(bit_sequence) < 16:
        return {}
    
    n = len(bit_sequence)
    max_period = min(n // 4, 256)  # Check periods up to n/4 or 256
    
    periodic_scores = {}
    
    for period in range(2, max_period + 1):
        matches = 0
        comparisons = 0
        
        # Check how well the sequence repeats with this period
        for i in range(period, n):
            if bit_sequence[i] == bit_sequence[i % period]:
                matches += 1
            comparisons += 1
        
        if comparisons > 0:
            score = matches / comparisons
            periodic_scores[period] = score
    
    # Find strongest periodicities
    strong_periods = [(period, score) for period, score in periodic_scores.items() if score > 0.7]
    strong_periods.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'periodic_scores': periodic_scores,
        'strong_periods': strong_periods[:10],  # Top 10 periodic patterns
        'max_periodicity': max(periodic_scores.values()) if periodic_scores else 0,
        'avg_periodicity': sum(periodic_scores.values()) / len(periodic_scores) if periodic_scores else 0
    }

def analyze_spatial_correlation(pixel_data):
    """Analyze spatial correlation in LSB patterns."""
    if not pixel_data or 'pixels' not in pixel_data:
        return {}
    
    width = pixel_data['width']
    height = pixel_data['height']
    pixels = pixel_data['pixels']
    
    if width < 2 or height < 2:
        return {}
    
    # Extract LSBs in 2D grid format
    red_grid = []
    green_grid = []
    blue_grid = []
    
    for y in range(height):
        red_row = []
        green_row = []
        blue_row = []
        
        for x in range(width):
            pixel_idx = y * width + x
            if pixel_idx < len(pixels):
                r, g, b = pixels[pixel_idx]
                red_row.append(r & 1)
                green_row.append(g & 1)
                blue_row.append(b & 1)
        
        red_grid.append(red_row)
        green_grid.append(green_row)
        blue_grid.append(blue_row)
    
    # Calculate horizontal and vertical correlations
    results = {}
    
    for channel_name, grid in [('red', red_grid), ('green', green_grid), ('blue', blue_grid)]:
        horizontal_corr = calculate_2d_correlation(grid, direction='horizontal')
        vertical_corr = calculate_2d_correlation(grid, direction='vertical')
        diagonal_corr = calculate_2d_correlation(grid, direction='diagonal')
        
        results[channel_name] = {
            'horizontal_correlation': horizontal_corr,
            'vertical_correlation': vertical_corr,
            'diagonal_correlation': diagonal_corr
        }
    
    return results

def calculate_2d_correlation(grid, direction='horizontal'):
    """Calculate 2D correlation in specified direction."""
    if not grid or len(grid) < 2:
        return 0
    
    correlations = []
    
    if direction == 'horizontal':
        for row in grid:
            if len(row) >= 2:
                for i in range(len(row) - 1):
                    correlations.append(1 if row[i] == row[i + 1] else -1)
    
    elif direction == 'vertical':
        for i in range(len(grid) - 1):
            for j in range(min(len(grid[i]), len(grid[i + 1]))):
                correlations.append(1 if grid[i][j] == grid[i + 1][j] else -1)
    
    elif direction == 'diagonal':
        for i in range(len(grid) - 1):
            for j in range(min(len(grid[i]), len(grid[i + 1])) - 1):
                correlations.append(1 if grid[i][j] == grid[i + 1][j + 1] else -1)
    
    return sum(correlations) / len(correlations) if correlations else 0

def perform_runs_test(bit_sequence):
    """Perform runs test for randomness."""
    if not bit_sequence or len(bit_sequence) < 2:
        return {}
    
    n = len(bit_sequence)
    n0 = bit_sequence.count(0)
    n1 = bit_sequence.count(1)
    
    if n0 == 0 or n1 == 0:
        return {'runs_test_result': 'degenerate'}
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if bit_sequence[i] != bit_sequence[i - 1]:
            runs += 1
    
    # Expected number of runs for random sequence
    expected_runs = (2 * n0 * n1) / n + 1
    
    # Variance of runs for random sequence
    runs_variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    
    # Z-score for runs test
    if runs_variance > 0:
        z_score = (runs - expected_runs) / (runs_variance ** 0.5)
    else:
        z_score = 0
    
    # Critical values for 95% confidence (approximately ±1.96)
    is_random = abs(z_score) < 1.96
    
    return {
        'total_runs': runs,
        'expected_runs': expected_runs,
        'runs_variance': runs_variance,
        'z_score': z_score,
        'is_random': is_random,
        'randomness_score': max(0, 1 - abs(z_score) / 3)  # Normalized score
    }

def analyze_frequency_domain(bit_sequence):
    """Analyze frequency domain characteristics of LSB data."""
    if not bit_sequence or len(bit_sequence) < 8:
        return {}
    
    n = len(bit_sequence)
    
    # Simple DFT-like analysis without imports
    # Focus on detecting low-frequency patterns
    frequencies = []
    
    # Check for patterns at different frequencies
    for freq in range(1, min(n // 2, 64)):  # Limit to reasonable range
        real_sum = 0
        imag_sum = 0
        
        for k in range(n):
            angle = 2 * 3.14159 * freq * k / n
            cos_val = cos_approx(angle)
            sin_val = sin_approx(angle)
            
            bit_val = 1 if bit_sequence[k] else -1
            real_sum += bit_val * cos_val
            imag_sum += bit_val * sin_val
        
        magnitude = (real_sum * real_sum + imag_sum * imag_sum) ** 0.5
        frequencies.append((freq, magnitude))
    
    # Find dominant frequencies
    frequencies.sort(key=lambda x: x[1], reverse=True)
    dominant_freqs = frequencies[:5]
    
    # Calculate spectral flatness (measure of randomness)
    magnitudes = [mag for _, mag in frequencies if mag > 0]
    if magnitudes:
        geometric_mean = 1
        for mag in magnitudes:
            geometric_mean *= mag ** (1 / len(magnitudes))
        
        arithmetic_mean = sum(magnitudes) / len(magnitudes)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    else:
        spectral_flatness = 0
    
    return {
        'dominant_frequencies': dominant_freqs,
        'spectral_flatness': spectral_flatness,
        'frequency_distribution': frequencies
    }

def cos_approx(x):
    """Approximate cosine using Taylor series."""
    # Normalize x to [0, 2π]
    while x > 6.28318:
        x -= 6.28318
    while x < 0:
        x += 6.28318
    
    # Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    result = 1
    term = 1
    x_squared = x * x
    
    for i in range(1, 10):  # Use first 10 terms
        term *= -x_squared / ((2 * i - 1) * (2 * i))
        result += term
        if abs(term) < 1e-10:  # Convergence check
            break
    
    return result

def sin_approx(x):
    """Approximate sine using Taylor series."""
    # Normalize x to [0, 2π]
    while x > 6.28318:
        x -= 6.28318
    while x < 0:
        x += 6.28318
    
    # Taylor series: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    result = x
    term = x
    x_squared = x * x
    
    for i in range(1, 10):  # Use first 10 terms
        term *= -x_squared / ((2 * i) * (2 * i + 1))
        result += term
        if abs(term) < 1e-10:  # Convergence check
            break
    
    return result

def calculate_complexity_metrics(bit_sequence):
    """Calculate various complexity metrics for LSB data."""
    if not bit_sequence:
        return {}
    
    n = len(bit_sequence)
    
    # Lempel-Ziv complexity approximation
    lz_complexity = approximate_lz_complexity(bit_sequence)
    
    # Binary derivative (XOR with shifted version)
    binary_derivative = []
    for i in range(1, n):
        binary_derivative.append(bit_sequence[i] ^ bit_sequence[i - 1])
    
    # Second-order derivative
    second_derivative = []
    for i in range(1, len(binary_derivative)):
        second_derivative.append(binary_derivative[i] ^ binary_derivative[i - 1])
    
    # Calculate metrics for derivatives
    first_order_entropy = calculate_entropy(binary_derivative)
    second_order_entropy = calculate_entropy(second_derivative)
    
    return {
        'lz_complexity': lz_complexity,
        'normalized_lz': lz_complexity / n if n > 0 else 0,
        'first_order_entropy': first_order_entropy,
        'second_order_entropy': second_order_entropy,
        'binary_derivative': binary_derivative,
        'complexity_score': (lz_complexity / n + first_order_entropy + second_order_entropy) / 3 if n > 0 else 0
    }

def approximate_lz_complexity(bit_sequence):
    """Approximate Lempel-Ziv complexity."""
    if not bit_sequence:
        return 0
    
    complexity = 1
    i = 0
    n = len(bit_sequence)
    
    while i < n:
        max_match = 0
        
        # Find longest match in previous subsequence
        for k in range(i):
            match_length = 0
            while (k + match_length < i and 
                   i + match_length < n and 
                   bit_sequence[k + match_length] == bit_sequence[i + match_length]):
                match_length += 1
            max_match = max(max_match, match_length)
        
        # Move to next unmatched position
        i += max(1, max_match)
        complexity += 1
    
    return complexity

def detect_stego(image_path):
    """Main function to detect stego content using LSB analysis."""
    print(f"Starting analysis of: {image_path}")
    
    # Read BMP file
    pixel_data = read_bmp_pixels(image_path)
    if not pixel_data:
        return {
            'error': 'Failed to read image file',
            'is_stego': False,
            'confidence': 0
        }
    
    print(f"Image loaded: {pixel_data['width']}x{pixel_data['height']} pixels")
    
    # Extract LSB plane
    lsb_plane = extract_lsb_plane(pixel_data)
    if not lsb_plane:
        return {
            'error': 'Failed to extract LSB plane',
            'is_stego': False,
            'confidence': 0
        }
    
    print("LSB extraction complete")
    
    # Perform comprehensive analysis
    results = {
        'image_info': {
            'width': pixel_data['width'],
            'height': pixel_data['height'],
            'total_pixels': pixel_data['total_pixels']
        }
    }
    
    # Basic bit distribution analysis
    print("Analyzing bit distribution...")
    bit_distribution = analyze_bit_distribution(lsb_plane)
    results['bit_distribution'] = bit_distribution
    
    # Analyze combined LSB data
    combined_lsbs = lsb_plane['combined_lsbs']
    
    # Entropy analysis
    print("Calculating entropy...")
    entropy = calculate_entropy(combined_lsbs)
    results['entropy'] = entropy
    
    # Sequential pattern analysis
    print("Analyzing sequential patterns...")
    sequential_patterns = analyze_sequential_patterns(combined_lsbs)
    results['sequential_patterns'] = sequential_patterns
    
    # Block pattern analysis
    print("Analyzing block patterns...")
    block_patterns_8 = analyze_block_patterns(combined_lsbs, 8)
    block_patterns_16 = analyze_block_patterns(combined_lsbs, 16)
    results['block_patterns'] = {
        'block_8': block_patterns_8,
        'block_16': block_patterns_16
    }
    
    # Autocorrelation analysis
    print("Calculating autocorrelation...")
    autocorr = calculate_autocorrelation(combined_lsbs, 50)
    results['autocorrelation'] = autocorr
    
    # Periodic pattern detection
    print("Detecting periodic patterns...")
    periodic_patterns = detect_periodic_patterns(combined_lsbs)
    results['periodic_patterns'] = periodic_patterns
    
    # Spatial correlation analysis
    print("Analyzing spatial correlation...")
    spatial_correlation = analyze_spatial_correlation(pixel_data)
    results['spatial_correlation'] = spatial_correlation
    
    # Runs test for randomness
    print("Performing runs test...")
    runs_test = perform_runs_test(combined_lsbs)
    results['runs_test'] = runs_test
    
    # Frequency domain analysis
    print("Analyzing frequency domain...")
    freq_analysis = analyze_frequency_domain(combined_lsbs)
    results['frequency_analysis'] = freq_analysis
    
    # Complexity metrics
    print("Calculating complexity metrics...")
    complexity = calculate_complexity_metrics(combined_lsbs)
    results['complexity'] = complexity
    
    # Channel-specific analysis
    print("Analyzing individual channels...")
    channel_analysis = {}
    for channel in ['red_lsbs', 'green_lsbs', 'blue_lsbs']:
        if channel in lsb_plane:
            channel_bits = lsb_plane[channel]
            channel_analysis[channel] = {
                'entropy': calculate_entropy(channel_bits),
                'runs_test': perform_runs_test(channel_bits),
                'sequential_patterns': analyze_sequential_patterns(channel_bits),
                'periodic_patterns': detect_periodic_patterns(channel_bits)
            }
    
    results['channel_analysis'] = channel_analysis
    
    # Calculate detection scores and final decision
    print("Calculating detection score...")
    detection_result = calculate_detection_score(results)
    results.update(detection_result)
    
    print("Analysis complete!")
    return results

def calculate_detection_score(analysis_results):
    """Calculate final detection score based on all analyses."""
    scores = []
    weights = []
    
    # Entropy-based detection
    if 'entropy' in analysis_results:
        entropy = analysis_results['entropy']
        # High entropy (close to 1) suggests random data
        #entropy_score = min(entropy, 1.0) if entropy > 0.8 else 0
        if entropy > 0.7:
            entropy_score = 0.9
        elif entropy > 0.68:
            entropy_score = 0.7
        elif entropy > 0.65:
            entropy_score = 0.5
        elif entropy < 0.6:
            entropy_score = 0.1
        else:
            entropy_score = 0.3                
        scores.append(entropy_score)
        # High weights for entropy
        weights.append(0.30)
    
    # Bit distribution analysis
    if 'bit_distribution' in analysis_results and 'combined_lsbs' in analysis_results['bit_distribution']:
        dist = analysis_results['bit_distribution']['combined_lsbs']
        balance_score = dist.get('balance_score', 0)
        chi_square = dist.get('chi_square', 0)
        
        # Suspicious if too balanced (close to 0.5/0.5 split)
       #balance_suspicion = 1 - min(balance_score * 10, 1.0)  # Inverted for suspicion
        #chi_suspicion = min(chi_square / 100, 1.0)  # High chi-square is suspicious               
        if chi_square < 500:
            chi_suspicion = 0.95 
        elif chi_square < 1500:
            chi_suspicion = 0.8
        elif chi_square < 5000:
            chi_suspicion = 0.6
        elif chi_square < 10000:
            chi_suspicion = 0.4
        else:
            chi_suspicion = 0.05  
        
        # Bit balance analysis
        if balance_score < 0.05: 
            balance_suspicion = 0.9
        elif balance_score < 0.1:
            balance_suspicion = 0.7
        elif balance_score < 0.15:
            balance_suspicion = 0.5
        else:
            balance_suspicion = 0.1
                           
        scores.extend([balance_suspicion, chi_suspicion])
        weights.extend([0.35, 0.2])                   
    # Sequential patterns
    if 'sequential_patterns' in analysis_results:
        seq = analysis_results['sequential_patterns']
        transition_rate = seq.get('transition_rate', 0.5)
        
        # High transition rate suggests randomness (suspicious)
        #transition_suspicion = min(transition_rate * 2, 1.0)
        if transition_rate > 0.14:
            transition_suspicion = 0.8
        elif transition_rate > 0.13:
            transition_suspicion = 0.6
        elif transition_rate > 0.125:
            transition_suspicion = 0.4
        else:
            transition_suspicion = 0.2
        scores.append(transition_suspicion)
        weights.append(0.1)
    
    # Block patterns
    if 'block_patterns' in analysis_results:
        block_8 = analysis_results['block_patterns'].get('block_8', {})
        pattern_diversity = block_8.get('pattern_diversity', 0)
        block_entropy = block_8.get('block_entropy', 0)
        
        # High diversity and entropy suggest embedded data
        diversity_score = min(pattern_diversity * 1.5, 1.0)
        entropy_score = min(block_entropy / 8, 1.0)
        
        scores.extend([diversity_score, entropy_score])
        weights.extend([0.08, 0.08])
    
    # Periodic patterns
    if 'periodic_patterns' in analysis_results:
        periodic = analysis_results['periodic_patterns']
        max_periodicity = periodic.get('max_periodicity', 0)
        
        # Low periodicity suggests randomness (suspicious)
        periodicity_score = 1 - max_periodicity
        scores.append(periodicity_score)
        weights.append(0.1)
    
    # Runs test
    if 'runs_test' in analysis_results:
        runs = analysis_results['runs_test']
        #randomness_score = runs.get('randomness_score', 0)
        z_score = runs.get('z_score', 0)
        if z_score > -310:  
            z_score_suspicion = 0.7
        elif z_score > -320:
            z_score_suspicion = 0.5
        else:
            z_score_suspicion = 0.2
        # High randomness is suspicious
        #scores.append(randomness_score)
        scores.append(z_score_suspicion)
        weights.append(0.05)
    
    # Spatial correlation
    if 'spatial_correlation' in analysis_results:
        spatial_scores = []
        for channel in ['red', 'green', 'blue']:
            if channel in analysis_results['spatial_correlation']:
                corr_data = analysis_results['spatial_correlation'][channel]
                avg_corr = (abs(corr_data.get('horizontal_correlation', 0)) + 
                           abs(corr_data.get('vertical_correlation', 0)) + 
                           abs(corr_data.get('diagonal_correlation', 0))) / 3
                
                # Low spatial correlation suggests embedded data
                spatial_scores.append(max(0, 1 - avg_corr))
        
        if spatial_scores:
            avg_spatial_score = sum(spatial_scores) / len(spatial_scores)
            scores.append(avg_spatial_score)
            weights.append(0.1)
    
    # Complexity metrics
    if 'complexity' in analysis_results:
        complexity = analysis_results['complexity']
        complexity_score = complexity.get('complexity_score', 0)
        
        # High complexity suggests embedded data
        scores.append(min(complexity_score, 1.0))
        weights.append(0.08)
    
    # Channel consistency analysis
    if 'channel_analysis' in analysis_results:
        channel_entropies = []
        for channel in ['red_lsbs', 'green_lsbs', 'blue_lsbs']:
            if channel in analysis_results['channel_analysis']:
                entropy = analysis_results['channel_analysis'][channel].get('entropy', 0)
                channel_entropies.append(entropy)
        
        if len(channel_entropies) >= 2:
            # Calculate variance in channel entropies
            mean_entropy = sum(channel_entropies) / len(channel_entropies)
            variance = sum((e - mean_entropy) ** 2 for e in channel_entropies) / len(channel_entropies)
            
            # High variance suggests selective channel embedding
            consistency_score = min(variance * 20, 1.0)
            scores.append(consistency_score)
            weights.append(0.07)
    
    # Calculate weighted average
    if scores and weights:
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            final_score = sum(scores) / len(scores)
    else:
        final_score = 0
    
    # Determine classification
    confidence = min(max(final_score, 0), 1)
    
    # Threshold-based classification
    if confidence > 0.75:
        classification = "highly_likely_stego"
        is_stego = True
    elif confidence > 0.6:
        classification = "likely_stego"
        is_stego = True
    elif confidence > 0.4:
        classification = "suspicious"
        is_stego = True
    else:
        classification = "likely_clean"
        is_stego = False
    
    return {
        'is_stego': is_stego,
        'confidence': confidence,
        'classification': classification,
        'detection_scores': {
            'individual_scores': scores,
            'weights': weights,
            'final_score': final_score
        }
    }

def generate_detection_report(analysis_results):
    """Generate a comprehensive detection report."""
    if not analysis_results:
        return "No analysis results available."
    
    report = []
    report.append("=== LSB Steganography Detection Report ===\n")
    
    # Image information
    if 'image_info' in analysis_results:
        info = analysis_results['image_info']
        report.append(f"Image Dimensions: {info['width']}x{info['height']}")
        report.append(f"Total Pixels: {info['total_pixels']}")
        report.append("")
    
    # Detection result
    is_stego = analysis_results.get('is_stego', False)
    confidence = analysis_results.get('confidence', 0)
    classification = analysis_results.get('classification', 'unknown')
    
    report.append(f"Detection Result: {'STEGANOGRAPHY DETECTED' if is_stego else 'LIKELY CLEAN'}")
    report.append(f"Confidence: {confidence:.3f}")
    report.append(f"Classification: {classification}")
    report.append("")
    
    # Detailed analysis
    if 'entropy' in analysis_results:
        report.append(f"Overall Entropy: {analysis_results['entropy']:.4f}")
    
    if 'bit_distribution' in analysis_results and 'combined_lsbs' in analysis_results['bit_distribution']:
        dist = analysis_results['bit_distribution']['combined_lsbs']
        report.append(f"Bit Balance (0s/1s): {dist['ratio_0']:.3f}/{dist['ratio_1']:.3f}")
        report.append(f"Chi-square: {dist['chi_square']:.3f}")
    
    if 'runs_test' in analysis_results:
        runs = analysis_results['runs_test']
        report.append(f"Runs Test Z-score: {runs.get('z_score', 0):.3f}")
        report.append(f"Randomness Score: {runs.get('randomness_score', 0):.3f}")
    
    if 'sequential_patterns' in analysis_results:
        seq = analysis_results['sequential_patterns']
        report.append(f"Transition Rate: {seq.get('transition_rate', 0):.3f}")
    
    if 'periodic_patterns' in analysis_results:
        periodic = analysis_results['periodic_patterns']
        report.append(f"Max Periodicity: {periodic.get('max_periodicity', 0):.3f}")
    
    if 'complexity' in analysis_results:
        complexity = analysis_results['complexity']
        report.append(f"Complexity Score: {complexity.get('complexity_score', 0):.3f}")
    
    report.append("")
    report.append("=== End Report ===")
    
    return "\n".join(report)

def main():
    """Main function for command-line interface."""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file analysis: python lsb_analyser.py <image_path>")
        print("  Examples:")
        print("    python lsb_analyser.py data/datasets/cover/image1.bmp")
        print("    python lsb_analyser.py data/datasets/cover results_cover.csv")
        print("    python lsb_analyser.py data/datasets/stego results_stego.csv")
        return
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        return
    
    if os.path.isfile(input_path):
        # Single file analysis
        print(f"Analyzing single file: {input_path}")
        result = detect_stego(input_path)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Generate and display report
        report = generate_detection_report(result)
        print("\n" + report)
        
        # Also print key metrics
        print("\nKey Metrics:")
        print(f"  File: {os.path.basename(input_path)}")
        print(f"  Is Stego: {result['is_stego']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Classification: {result['classification']}")
        print(f"  Entropy: {result.get('entropy', 0):.4f}")
        
        if 'bit_distribution' in result and 'combined_lsbs' in result['bit_distribution']:
            dist = result['bit_distribution']['combined_lsbs']
            print(f"  Chi-square: {dist.get('chi_square', 0):.4f}")
        
        if 'runs_test' in result:
            runs = result['runs_test']
            print(f"  Randomness Score: {runs.get('randomness_score', 0):.4f}")
    
    else:
        print("error")

if __name__ == "__main__":
    main()