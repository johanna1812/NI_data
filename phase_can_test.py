#%%
import os
import glob
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import find_peaks
import warnings
from scipy.optimize import curve_fit
import matplotlib.lines as mlines

# ==========================================
# 1. PARSING & MATRIX MATH HELPER FUNCTIONS
# ==========================================
def get_detector2_data_from_file(file_path):
    """
    Reads one file and returns:
    1. A list of dicts for each HERALD Detector 2 event (matrix + n_sub).
    2. The marginal joint statistics (sum of all Det 2 heralded matrices).
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = content.split("HERALD:")
    heralded_results = []
    
    for block in blocks[1:]:
        block_text = "HERALD:" + block
        lines = block_text.split('\n')
        
        herald_match = re.search(r"HERALD: Detector (\d+).*?Px (\d+)", lines[0])
        
        if herald_match:
            det_num = herald_match.group(1)
            # We ONLY care about Detector 2
            if det_num == '1':
                n_sub = int(herald_match.group(2)) - 1
                
                matrix_rows = []
                for line in lines:
                    if line.strip().startswith("Px") and ":" in line:
                        data_part = line.split(":")[1]
                        vals = re.findall(r"[-+]?\d*\.\d+|\d+", data_part)
                        if vals:
                            matrix_rows.append([float(v) for v in vals])
                
                if matrix_rows:
                    # Ensure rectangular/square matrix
                    row_len = len(matrix_rows[0])
                    valid_rows = [r for r in matrix_rows if len(r) == row_len]
                    matrix = np.array(valid_rows)
                    heralded_results.append({
                        "n_sub": n_sub,
                        "matrix": matrix
                    })

    if not heralded_results:
        return None, None

    marginal_matrix = sum(item['matrix'] for item in heralded_results)
    return heralded_results, marginal_matrix


def pad_and_add(A, B):
    """Safely adds two matrices, padding with zeros if their dimensions differ."""
    if A is None: return B.copy() if B is not None else None
    if B is None: return A.copy()
    
    max_r = max(A.shape[0], B.shape[0])
    max_c = max(A.shape[1], B.shape[1])
    
    new_A = np.zeros((max_r, max_c))
    new_B = np.zeros((max_r, max_c))
    
    new_A[:A.shape[0], :A.shape[1]] = A
    new_B[:B.shape[0], :B.shape[1]] = B
    
    return new_A + new_B


def extract_marginals(matrix):
    """Normalizes the joint matrix and extracts 1D marginals for Det 3 and Det 4."""
    if matrix is None or np.sum(matrix) == 0:
        return None, None
    P_joint = matrix / np.sum(matrix)
    Pn_det3 = np.sum(P_joint, axis=1) # Sum across columns
    Pm_det4 = np.sum(P_joint, axis=0) # Sum across rows
    return Pn_det3, Pm_det4

def extract_normalized_joint(matrix):
    """Normalizes and returns the full 2D joint probability matrix P(n,m)."""
    if matrix is None or np.sum(matrix) == 0:
        return None
    return matrix / np.sum(matrix)

# ==========================================
# 2. MAIN PROCESSING FUNCTION
# ==========================================
def analyze_voltage_scan(main_folder):
    """
    Scans the main folder for 'voltage_X.XV_UP' and 'DOWN', processes File 1 
    and the Average of all files, and plots the P(n) trends over voltage.
    """
    print(f"Scanning {main_folder} for voltage folders...")
    
    # Structure: data[dataset_type][n_sub][voltage] = {'Det3': P_array, 'Det4': P_array}
    data = {
        'file1_UP': {}, 'avg_UP': {},
        'file1_DOWN': {}, 'avg_DOWN': {}
    }
    
    all_voltages = set()
    folder_map = {'UP': {}, 'DOWN': {}}

    # 1. Find and sort voltage folders, distinguishing between UP and DOWN
    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    
    for folder in subfolders:
        match = re.search(r"voltage_([0-9\.]+)V_(UP|DOWN)", folder, re.IGNORECASE)
        if match:
            v = float(match.group(1))
            direction = match.group(2).upper()
            folder_map[direction][v] = folder
            all_voltages.add(v)
            
    if not all_voltages:
        print("No valid 'voltage_X.XV_UP' or 'DOWN' folders found.")
        return

    # 2. Extract Data for both directions
    for direction in ['UP', 'DOWN']:
        voltages = sorted(list(folder_map[direction].keys()))
        if not voltages:
            continue
            
        print(f"Processing {len(voltages)} steps for {direction} sweep...")
        
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            if not os.path.exists(folder_path):
                continue
                
            txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            if not txt_files:
                continue
                
            ds_file1 = f'file1_{direction}'
            ds_avg = f'avg_{direction}'
            
            # --- PROCESS FILE 1 (Find the first valid text file) ---
            h_list_1, m_mat_1 = None, None
            for file in txt_files:
                h_list, m_mat = get_detector2_data_from_file(file)
                if m_mat is not None:
                    h_list_1, m_mat_1 = h_list, m_mat
                    break # Successfully found the first file with actual data!
            
            if m_mat_1 is not None:
                if -1 not in data[ds_file1]: data[ds_file1][-1] = {}
                p3, p4 = extract_marginals(m_mat_1)
                data[ds_file1][-1][v] = {'Det3': p3, 'Det4': p4}
                
                for item in h_list_1:
                    n_sub = item['n_sub']
                    if n_sub not in data[ds_file1]: data[ds_file1][n_sub] = {}
                    p3, p4 = extract_marginals(item['matrix'])
                    data[ds_file1][n_sub][v] = {'Det3': p3, 'Det4': p4}

            # --- PROCESS AVERAGE (ALL FILES) ---
            avg_m_mat = None
            avg_h_dict = {}
            
            for file in txt_files:
                h_list, m_mat = get_detector2_data_from_file(file)
                avg_m_mat = pad_and_add(avg_m_mat, m_mat)
                if h_list:
                    for item in h_list:
                        n_sub = item['n_sub']
                        avg_h_dict[n_sub] = pad_and_add(avg_h_dict.get(n_sub, None), item['matrix'])
            
            if avg_m_mat is not None:
                if -1 not in data[ds_avg]: data[ds_avg][-1] = {}
                p3, p4 = extract_marginals(avg_m_mat)
                data[ds_avg][-1][v] = {'Det3': p3, 'Det4': p4}
                
                for n_sub, mat in avg_h_dict.items():
                    if n_sub not in data[ds_avg]: data[ds_avg][n_sub] = {}
                    p3, p4 = extract_marginals(mat)
                    data[ds_avg][n_sub][v] = {'Det3': p3, 'Det4': p4}

    # 3. Plotting
    plot_voltage_trends(data, sorted(list(all_voltages)))


# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================
def plot_voltage_trends(data, all_voltages):
    """
    Creates separate figures for UP and DOWN sweeps. Inside each figure,
    generates a grid of subplots for each individual P(n).
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Find all n_sub categories present across all datasets
    n_subs = set()
    for ds in data.values():
        n_subs.update(ds.keys())
    n_subs = sorted(list(n_subs))

    # Iterate over sweeps separately to generate completely clean, distinct figures
    for direction in ['UP', 'DOWN']:
        ds_avg = f'avg_{direction}'
        ds_file1 = f'file1_{direction}'
        
        # Verify this direction has data before creating figures
        if ds_avg not in data and ds_file1 not in data:
            continue
        if not data[ds_avg] and not data[ds_file1]:
            continue

        for n_sub in n_subs:
            # 1. Determine the max_n for THIS direction and THIS n_sub
            max_n = 0
            for ds in [ds_avg, ds_file1]:
                if ds in data and n_sub in data[ds]:
                    for v_data in data[ds][n_sub].values():
                        for det in ['Det3', 'Det4']:
                            if v_data[det] is not None:
                                max_n = max(max_n, len(v_data[det]))
            # NEW: Cap the maximum photon number to n=5
            max_n = min(max_n, 6) 
            if max_n == 0:
                continue
                
            # 2. Setup the Subplot Grid
            cols = min(3, max_n)
            rows = math.ceil(max_n / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2.5), sharex=True)
            
            if max_n == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            title_str = "Unheralded (Raw)" if n_sub == -1 else f"Heralded $N_1 = {n_sub}$"
            fig.suptitle(f"Probability $P(n)$ vs. Voltage [{direction} Sweep]: {title_str}", 
                         fontsize=18, fontweight='bold', y=0.98)

            # 3. Define Distinct, High-Contrast Line Styles
            # Avg = Thick, lighter solid colors. File 1 = Thin, dark dashed lines strictly ON TOP with large crosses.
            styles = {
                (ds_avg, 'Det3'):   {'color': '#7fb3d5', 'marker': 'o', 'ls': '-',  'label': f'Average ({direction}) | Det 3', 'ms': 8, 'zorder': 4, 'lw': 2.5, 'alpha': 0.8},
                (ds_avg, 'Det4'):   {'color': '#e6b0aa', 'marker': 's', 'ls': '-',  'label': f'Average ({direction}) | Det 4', 'ms': 8, 'zorder': 4, 'lw': 2.5, 'alpha': 0.8},
                (ds_file1, 'Det3'): {'color': '#154360', 'marker': 'x', 'ls': '--', 'label': f'File 1 ({direction}) | Det 3',  'ms': 10, 'zorder': 5, 'lw': 1.5, 'alpha': 1.0},
                (ds_file1, 'Det4'): {'color': '#641e16', 'marker': '+', 'ls': ':',  'label': f'File 1 ({direction}) | Det 4',  'ms': 14, 'zorder': 5, 'lw': 1.5, 'alpha': 1.0}
            }

            # 4. Plot each individual P(n) in its own subplot
            for k in range(max_n):
                ax = axes[k]
                
                for (ds, det), style in styles.items():
                    if ds not in data or n_sub not in data[ds]: continue
                    
                    v_dict = data[ds][n_sub]
                    valid_vs = sorted(list(v_dict.keys()))
                    
                    x_vals = []
                    y_vals = []
                    for v in valid_vs:
                        p_array = v_dict[v][det]
                        if p_array is not None and k < len(p_array):
                            x_vals.append(v)
                            y_vals.append(p_array[k])
                    
                    if y_vals and sum(y_vals) > 1e-6:
                        ax.plot(x_vals, y_vals, color=style['color'], marker=style['marker'], markersize=style['ms'],
                                linestyle=style['ls'], linewidth=style['lw'], alpha=style['alpha'], zorder=style['zorder'], label=style['label'])
                
                ax.set_title(f"$P(n={k})$", pad=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Formatting labels for the grid edges
                if k % cols == 0:
                    ax.set_ylabel("Probability")
                if k >= max_n - cols:
                    ax.set_xlabel("Voltage (V)")

            # 5. Hide any unused/empty subplots (if max_n isn't a multiple of cols)
            for k in range(max_n, len(axes)):
                fig.delaxes(axes[k])

            # 6. Unified Legend at the bottom (Robustly scan all subplots to ensure nothing is missed)
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
                        
            if handles:
                fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

            plt.tight_layout()
            # Make extra room for the legend at the bottom
            plt.subplots_adjust(bottom=0.20, top=0.9)
            plt.show()
#%%
def analyze_voltage_scan_all(main_folder):
    """
    Scans the main folder for 'voltage_X.XV_UP' and 'DOWN', processes ALL individual files 
    alongside the Average, and plots them with the individual files faintly in the background.
    """
    print(f"Scanning {main_folder} for voltage folders (All Files mode)...")
    
    # Structure: data[dataset_type][n_sub][voltage] = [{'Det3': P_array, 'Det4': P_array}, ...]
    data = {
        'all_UP': {}, 'avg_UP': {},
        'all_DOWN': {}, 'avg_DOWN': {}
    }
    
    all_voltages = set()
    folder_map = {'UP': {}, 'DOWN': {}}

    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    for folder in subfolders:
        match = re.search(r"voltage_([0-9\.]+)V_(UP|DOWN)", folder, re.IGNORECASE)
        if match:
            v = float(match.group(1))
            direction = match.group(2).upper()
            folder_map[direction][v] = folder
            all_voltages.add(v)
            
    if not all_voltages:
        print("No valid 'voltage_X.XV_UP' or 'DOWN' folders found.")
        return

    for direction in ['UP', 'DOWN']:
        voltages = sorted(list(folder_map[direction].keys()))
        if not voltages: continue
            
        print(f"Processing {len(voltages)} steps for {direction} sweep...")
        ds_all = f'all_{direction}'
        ds_avg = f'avg_{direction}'
        
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            if not os.path.exists(folder_path): continue
                
            txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            if not txt_files: continue
            
            avg_m_mat = None
            avg_h_dict = {}
            
            # --- PROCESS ALL FILES ---
            for file_idx, file in enumerate(txt_files):
                h_list, m_mat = get_detector2_data_from_file(file)
                
                if m_mat is not None:
                    # Accumulate for Average
                    avg_m_mat = pad_and_add(avg_m_mat, m_mat)
                    
                    # Store Individual File
                    if -1 not in data[ds_all]: data[ds_all][-1] = {}
                    if v not in data[ds_all][-1]: data[ds_all][-1][v] = []
                    p3, p4 = extract_marginals(m_mat)
                    data[ds_all][-1][v].append({'Det3': p3, 'Det4': p4})
                    
                if h_list:
                    for item in h_list:
                        n_sub = item['n_sub']
                        # Accumulate for Average
                        avg_h_dict[n_sub] = pad_and_add(avg_h_dict.get(n_sub, None), item['matrix'])
                        
                        # Store Individual File
                        if n_sub not in data[ds_all]: data[ds_all][n_sub] = {}
                        if v not in data[ds_all][n_sub]: data[ds_all][n_sub][v] = []
                        p3, p4 = extract_marginals(item['matrix'])
                        data[ds_all][n_sub][v].append({'Det3': p3, 'Det4': p4})
            
            # --- SAVE AVERAGE ---
            if avg_m_mat is not None:
                if -1 not in data[ds_avg]: data[ds_avg][-1] = {}
                p3, p4 = extract_marginals(avg_m_mat)
                data[ds_avg][-1][v] = {'Det3': p3, 'Det4': p4}
                
                for n_sub, mat in avg_h_dict.items():
                    if n_sub not in data[ds_avg]: data[ds_avg][n_sub] = {}
                    p3, p4 = extract_marginals(mat)
                    data[ds_avg][n_sub][v] = {'Det3': p3, 'Det4': p4}

    plot_voltage_trends_all(data, sorted(list(all_voltages)))


def plot_voltage_trends_all(data, all_voltages):
    """
    Plots the "spaghetti" lines for all individual files faintly in the background, 
    with the bold Average line overlaid on top. Separates UP and DOWN.
    """
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

    n_subs = set()
    for ds in data.values(): n_subs.update(ds.keys())
    n_subs = sorted(list(n_subs))

    for direction in ['UP', 'DOWN']:
        ds_avg = f'avg_{direction}'
        ds_all = f'all_{direction}'
        
        if ds_avg not in data and ds_all not in data: continue
        if not data[ds_avg] and not data[ds_all]: continue

        for n_sub in n_subs:
            max_n = 0
            if ds_avg in data and n_sub in data[ds_avg]:
                for v_data in data[ds_avg][n_sub].values():
                    for det in ['Det3', 'Det4']:
                        if v_data[det] is not None:
                            max_n = max(max_n, len(v_data[det]))
            # NEW: Cap the maximum photon number to n=5
            max_n = min(max_n, 6) 
            if max_n == 0: continue
                
            cols = min(3, max_n)
            rows = math.ceil(max_n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2.5), sharex=True)
            if max_n == 1: axes = [axes]
            else: axes = axes.flatten()

            title_str = "Unheralded (Raw)" if n_sub == -1 else f"Heralded $N_1 = {n_sub}$"
            fig.suptitle(f"Probability $P(n)$ vs. Voltage [{direction} Sweep]: {title_str}\n(All Individual Files + Average)", 
                         fontsize=18, fontweight='bold', y=0.98)

            # --- STYLES: Faint Individual Lines vs Thick Average Lines ---
            style_avg = {
                'Det3': {'color': '#154360', 'marker': 'o', 'ls': '-',  'label': f'Average | Det 3', 'ms': 7, 'zorder': 10, 'lw': 3, 'alpha': 1.0},
                'Det4': {'color': '#641e16', 'marker': 's', 'ls': '-',  'label': f'Average | Det 4', 'ms': 7, 'zorder': 10, 'lw': 3, 'alpha': 1.0}
            }
            style_ind = {
                'Det3': {'color': '#154360', 'marker': '.', 'ls': '-', 'label': f'Individual | Det 3', 'lw': 1.0, 'alpha': 0.75, 'zorder': 5},
                'Det4': {'color': '#641e16', 'marker': '.', 'ls': '-', 'label': f'Individual | Det 4', 'lw': 1.0, 'alpha': 0.75, 'zorder': 5}
            }

            for k in range(max_n):
                ax = axes[k]
                
                # 1. Plot Individual Files (Spaghetti Lines)
                if ds_all in data and n_sub in data[ds_all]:
                    v_dict_all = data[ds_all][n_sub]
                    valid_vs_all = sorted(list(v_dict_all.keys()))
                    
                    # Find maximum amount of files recorded at any voltage point
                    max_files = max([len(v_dict_all[v]) for v in valid_vs_all])
                    
                    for det in ['Det3', 'Det4']:
                        added_label = False
                        for file_idx in range(max_files):
                            x_vals, y_vals = [], []
                            for v in valid_vs_all:
                                if file_idx < len(v_dict_all[v]):
                                    p_array = v_dict_all[v][file_idx][det]
                                    if p_array is not None and k < len(p_array):
                                        x_vals.append(v)
                                        y_vals.append(p_array[k])
                                        
                            if y_vals and sum(y_vals) > 0:
                                # Only attach a legend label if we haven't already for this detector in this subplot
                                lbl = style_ind[det]['label'] if not added_label else ""
                                added_label = True
                                ax.plot(x_vals, y_vals, color=style_ind[det]['color'], marker=style_ind[det]['marker'],
                                        markersize=3, linestyle=style_ind[det]['ls'], 
                                        linewidth=style_ind[det]['lw'], alpha=style_ind[det]['alpha'], 
                                        zorder=style_ind[det]['zorder'], label=lbl)

                # 2. Plot Average Line (Bold and on top)
                if ds_avg in data and n_sub in data[ds_avg]:
                    v_dict_avg = data[ds_avg][n_sub]
                    valid_vs_avg = sorted(list(v_dict_avg.keys()))
                    
                    for det in ['Det3', 'Det4']:
                        x_vals, y_vals = [], []
                        for v in valid_vs_avg:
                            p_array = v_dict_avg[v][det]
                            if p_array is not None and k < len(p_array):
                                x_vals.append(v)
                                y_vals.append(p_array[k])
                                
                        if y_vals and sum(y_vals) > 0:
                            ax.plot(x_vals, y_vals, color=style_avg[det]['color'], marker=style_avg[det]['marker'], 
                                    markersize=style_avg[det]['ms'], linestyle=style_avg[det]['ls'], 
                                    linewidth=style_avg[det]['lw'], alpha=style_avg[det]['alpha'], 
                                    zorder=style_avg[det]['zorder'], label=style_avg[det]['label'])

                ax.set_title(f"$P(n={k})$", pad=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                if k % cols == 0: ax.set_ylabel("Probability")
                if k >= max_n - cols: ax.set_xlabel("Voltage (V)")

            # Hide empty subplots
            for k in range(max_n, len(axes)): fig.delaxes(axes[k])

            # Gather robust legends without empty labels
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label and label not in labels:
                        handles.append(handle)
                        labels.append(label)
                        
            if handles:
                fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20, top=0.9)
            plt.show()

def analyze_voltage_scan_coincidences(main_folder):
    """
    Scans the main folder for 'voltage_X.XV_UP' and 'DOWN', extracting the FULL 
    2D joint probability matrix P(n,m) for ALL files and the Average.
    """
    print(f"Scanning {main_folder} for voltage folders (Coincidence mode)...")
    
    # Structure: data[dataset_type][n_sub][voltage] = [P_joint_1, P_joint_2, ...] for 'all'
    #            data[dataset_type][n_sub][voltage] = P_joint_avg for 'avg'
    data = {
        'all_UP': {}, 'avg_UP': {},
        'all_DOWN': {}, 'avg_DOWN': {}
    }
    
    all_voltages = set()
    folder_map = {'UP': {}, 'DOWN': {}}

    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    for folder in subfolders:
        match = re.search(r"voltage_([0-9\.]+)V_(UP|DOWN)", folder, re.IGNORECASE)
        if match:
            v = float(match.group(1))
            direction = match.group(2).upper()
            folder_map[direction][v] = folder
            all_voltages.add(v)
            
    if not all_voltages:
        print("No valid 'voltage_X.XV_UP' or 'DOWN' folders found.")
        return

    for direction in ['UP', 'DOWN']:
        voltages = sorted(list(folder_map[direction].keys()))
        if not voltages: continue
            
        print(f"Processing {len(voltages)} steps for {direction} sweep...")
        ds_all = f'all_{direction}'
        ds_avg = f'avg_{direction}'
        
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            if not os.path.exists(folder_path): continue
                
            txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            if not txt_files: continue
            
            avg_m_mat = None
            avg_h_dict = {}
            
            # --- PROCESS ALL FILES ---
            for file in txt_files:
                h_list, m_mat = get_detector2_data_from_file(file)
                
                if m_mat is not None:
                    # Accumulate for Average
                    avg_m_mat = pad_and_add(avg_m_mat, m_mat)
                    
                    # Store Normalized Individual File
                    if -1 not in data[ds_all]: data[ds_all][-1] = {}
                    if v not in data[ds_all][-1]: data[ds_all][-1][v] = []
                    p_joint = extract_normalized_joint(m_mat)
                    data[ds_all][-1][v].append(p_joint)
                    
                if h_list:
                    for item in h_list:
                        n_sub = item['n_sub']
                        avg_h_dict[n_sub] = pad_and_add(avg_h_dict.get(n_sub, None), item['matrix'])
                        
                        if n_sub not in data[ds_all]: data[ds_all][n_sub] = {}
                        if v not in data[ds_all][n_sub]: data[ds_all][n_sub][v] = []
                        p_joint = extract_normalized_joint(item['matrix'])
                        data[ds_all][n_sub][v].append(p_joint)
            
            # --- SAVE NORMALIZED AVERAGE ---
            if avg_m_mat is not None:
                if -1 not in data[ds_avg]: data[ds_avg][-1] = {}
                data[ds_avg][-1][v] = extract_normalized_joint(avg_m_mat)
                
                for n_sub, mat in avg_h_dict.items():
                    if n_sub not in data[ds_avg]: data[ds_avg][n_sub] = {}
                    data[ds_avg][n_sub][v] = extract_normalized_joint(mat)

    plot_voltage_trends_coincidences(data, sorted(list(all_voltages)))


def plot_voltage_trends_coincidences(data, all_voltages):
    """
    Plots the joint statistics P(n, m). 
    Each subplot fixes n (Det 3). Inside each subplot, traces represent m=0,1,2,3,4 (Det 4).
    Individual files are faint lines; Averages are thick lines with markers.
    """
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    
    # We will use a standard color cycle for m = 0, 1, 2, 3, 4
    colors = cm.get_cmap('tab10')

    n_subs = set()
    for ds in data.values(): n_subs.update(ds.keys())
    n_subs = sorted(list(n_subs))

    for direction in ['UP', 'DOWN']:
        ds_avg = f'avg_{direction}'
        ds_all = f'all_{direction}'
        
        if ds_avg not in data and ds_all not in data: continue
        if not data[ds_avg] and not data[ds_all]: continue

        for n_sub in n_subs:
            # 1. Determine maximum dimensions of the joint matrices
            max_n = 0 # Max rows (Det 3)
            max_m = 0 # Max cols (Det 4)
            
            if ds_avg in data and n_sub in data[ds_avg]:
                for v_mat in data[ds_avg][n_sub].values():
                    if v_mat is not None:
                        max_n = max(max_n, v_mat.shape[0])
                        max_m = max(max_m, v_mat.shape[1])
            
            if max_n == 0: continue
            
            # Bound the plot grid so it doesn't get ridiculously large if there's an outlier n=10
            max_n_plot = min(max_n, 6) # Up to 6 subplots (n=0 to n=5)
            max_m_plot = min(max_m, 5) # Up to 5 lines per subplot (m=0 to m=4)
                
            cols = min(3, max_n_plot)
            rows = math.ceil(max_n_plot / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2.5), sharex=True)
            if max_n_plot == 1: axes = [axes]
            else: axes = axes.flatten()

            title_str = "Unheralded (Raw)" if n_sub == -1 else f"Heralded $N_1 = {n_sub}$"
            fig.suptitle(f"Joint Probability $P(n,m)$ vs. Voltage [{direction} Sweep]: {title_str}\n(Det 3 = $n$, Det 4 = $m$)", 
                         fontsize=18, fontweight='bold', y=0.98)

            # 2. Iterate over subplots (fixed n)
            for n in range(max_n_plot):
                ax = axes[n]
                
                # Iterate over lines (fixed m)
                for m in range(max_m_plot):
                    c = colors(m)
                    
                    # --- Plot Individual Files (Faint Lines) ---
                    if ds_all in data and n_sub in data[ds_all]:
                        v_dict_all = data[ds_all][n_sub]
                        valid_vs_all = sorted(list(v_dict_all.keys()))
                        
                        max_files = max([len(v_dict_all[v]) for v in valid_vs_all])
                        
                        for file_idx in range(max_files):
                            x_vals, y_vals = [], []
                            for v in valid_vs_all:
                                if file_idx < len(v_dict_all[v]):
                                    p_mat = v_dict_all[v][file_idx]
                                    # Ensure n and m are within the bounds of this specific file's matrix
                                    if p_mat is not None and n < p_mat.shape[0] and m < p_mat.shape[1]:
                                        x_vals.append(v)
                                        y_vals.append(p_mat[n, m])
                                        
                            if y_vals and sum(y_vals) > 1e-6:
                                ax.plot(x_vals, y_vals, color=c, linestyle='-', linewidth=1.0, alpha=0.25, zorder=5)

                    # --- Plot Average (Bold Line) ---
                    if ds_avg in data and n_sub in data[ds_avg]:
                        v_dict_avg = data[ds_avg][n_sub]
                        valid_vs_avg = sorted(list(v_dict_avg.keys()))
                        
                        x_vals, y_vals = [], []
                        for v in valid_vs_avg:
                            p_mat = v_dict_avg[v]
                            if p_mat is not None and n < p_mat.shape[0] and m < p_mat.shape[1]:
                                x_vals.append(v)
                                y_vals.append(p_mat[n, m])
                                
                        if y_vals and sum(y_vals) > 1e-6:
                            ax.plot(x_vals, y_vals, color=c, marker='o', markersize=5, linestyle='-', linewidth=2.5, alpha=1.0, zorder=10)

                ax.set_title(f"Det 3 fixed at $n={n}$", pad=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                if n % cols == 0: ax.set_ylabel(f"$P(n={n}, m)$")
                if n >= max_n_plot - cols: ax.set_xlabel("Voltage (V)")

            # Hide empty subplots
            for k in range(max_n_plot, len(axes)): fig.delaxes(axes[k])

            # 3. Create a Custom Legend to explain colors (m) and line styles (Avg vs Individual)
            custom_handles = []
            
            # Add color markers for 'm'
            for m in range(max_m_plot):
                custom_handles.append(mlines.Line2D([], [], color=colors(m), marker='o', linestyle='-', linewidth=2.5, label=f'Det 4 ($m={m}$)'))
                
            # Add a spacer (invisible line)
            custom_handles.append(mlines.Line2D([], [], color='none', label=' '))
            
            # Add style explanations
            custom_handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='-', linewidth=2.5, label='Average'))
            custom_handles.append(mlines.Line2D([], [], color='black', marker='', linestyle='-', linewidth=1.0, alpha=0.4, label='Individual File'))

            fig.legend(handles=custom_handles, loc='lower center', ncol=min(max_m_plot+3, 8), bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20, top=0.9)
            plt.show()


def analyze_voltage_scan_drift_compensated(main_folder):
    """
    Scans folders, aligns each individual sweep (file index) by finding the first minimum 
    of P(n=0) to compensate for phase drift, and plots the aligned individuals + average.
    """
    print(f"Scanning {main_folder} for voltage folders (Drift Compensation mode)...")
    
    # Structure: sweeps[direction][file_idx] = {'V': [], 'data': { n_sub: {'Det3': [P_arr], 'Det4': [P_arr]} }}
    sweeps = {'UP': [], 'DOWN': []}
    folder_map = {'UP': {}, 'DOWN': {}}

    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    for folder in subfolders:
        match = re.search(r"voltage_([0-9\.]+)V_(UP|DOWN)", folder, re.IGNORECASE)
        if match:
            v = float(match.group(1))
            direction = match.group(2).upper()
            folder_map[direction][v] = folder

    # 1. Parse Data into Sweeps
    for direction in ['UP', 'DOWN']:
        voltages = sorted(list(folder_map[direction].keys()))
        if not voltages: continue
        
        # Find maximum number of files (sweeps) recorded in this direction
        max_files = 0
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            max_files = max(max_files, len(txt_files))
            
        print(f"Processing {max_files} sweeps for {direction} direction across {len(voltages)} voltages...")
        
        for i in range(max_files):
            sweeps[direction].append({'V': [], 'data': {}})
            
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save2")
            if not os.path.exists(folder_path): continue
            
            txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            
            for i, file in enumerate(txt_files):
                h_list, m_mat = get_detector2_data_from_file(file)
                if m_mat is None: continue
                
                sweeps[direction][i]['V'].append(v)
                
                # Unheralded
                p3, p4 = extract_marginals(m_mat)
                if -1 not in sweeps[direction][i]['data']:
                    sweeps[direction][i]['data'][-1] = {'Det3': [], 'Det4': []}
                sweeps[direction][i]['data'][-1]['Det3'].append(p3)
                sweeps[direction][i]['data'][-1]['Det4'].append(p4)
                
                # Heralded
                if h_list:
                    for item in h_list:
                        n_sub = item['n_sub']
                        p3, p4 = extract_marginals(item['matrix'])
                        if n_sub not in sweeps[direction][i]['data']:
                            sweeps[direction][i]['data'][n_sub] = {'Det3': [], 'Det4': []}
                        sweeps[direction][i]['data'][n_sub]['Det3'].append(p3)
                        sweeps[direction][i]['data'][n_sub]['Det4'].append(p4)

    # 2. Calculate Phase Drifts (Shifts)
    shifts = {'UP': [], 'DOWN': []}
    for direction in ['UP', 'DOWN']:
        ref_vmin = None
        for i, sweep in enumerate(sweeps[direction]):
            if -1 not in sweep['data'] or len(sweep['V']) < 3:
                shifts[direction].append(0)
                continue
                
            V = np.array(sweep['V'])
            p3_list = sweep['data'][-1]['Det3']
            # Safely extract P(0)
            p0 = np.array([p[0] if p is not None and len(p) > 0 else 0 for p in p3_list])
            
            if sum(p0) == 0:
                shifts[direction].append(0)
                continue
                
            # Find the first minimum (inverted peaks)
            # Prominence ensures we find a real fringe valley, not statistical noise
            peaks, _ = find_peaks(-p0, distance=3, prominence=np.max(p0)*0.05)
            if len(peaks) > 0:
                v_min = V[peaks[0]]
            else:
                # Fallback to absolute minimum
                v_min = V[np.argmin(p0)]
                
            # We align everything to the minimum of the first valid sweep
            if ref_vmin is None:
                ref_vmin = v_min
                
            shift_val = ref_vmin - v_min
            shifts[direction].append(shift_val)
            
        if sweeps[direction]:
            print(f"[{direction}] Calculated voltage shifts: {[round(s, 3) for s in shifts[direction]]}")

    plot_voltage_trends_drift_compensated(sweeps, shifts)


def plot_voltage_trends_drift_compensated(sweeps, shifts):
    """
    Plots the drift-compensated sweeps.
    Individual lines are plotted on their shifted axes.
    The average is calculated via interpolation.
    """
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

    for direction in ['UP', 'DOWN']:
        if not sweeps[direction]: continue

        # Gather all n_subs present in this direction
        n_subs = set()
        for sw in sweeps[direction]:
            n_subs.update(sw['data'].keys())
        n_subs = sorted(list(n_subs))

        for n_sub in n_subs:
            # 1. Determine max_n for plotting grid
            max_n = 0
            for sw in sweeps[direction]:
                if n_sub in sw['data']:
                    for det in ['Det3', 'Det4']:
                        for p_arr in sw['data'][n_sub][det]:
                            if p_arr is not None:
                                max_n = max(max_n, len(p_arr))
            # NEW: Cap the maximum photon number to n=5
            max_n = min(max_n, 6) 
            if max_n == 0: continue
                
            cols = min(3, max_n)
            rows = math.ceil(max_n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2.5), sharex=True)
            if max_n == 1: axes = [axes]
            else: axes = axes.flatten()

            title_str = "Unheralded (Raw)" if n_sub == -1 else f"Heralded $N_1 = {n_sub}$"
            fig.suptitle(f"Phase-Compensated $P(n)$ vs. Voltage [{direction} Sweep]: {title_str}\n(Aligned to First Minimum)", 
                         fontsize=18, fontweight='bold', y=0.98)

            style_avg = {
                'Det3': {'color': '#154360', 'marker': 'o', 'ls': '-',  'label': f'Compensated Avg | Det 3', 'ms': 7, 'zorder': 10, 'lw': 3, 'alpha': 1.0},
                'Det4': {'color': '#641e16', 'marker': 's', 'ls': '-',  'label': f'Compensated Avg | Det 4', 'ms': 7, 'zorder': 10, 'lw': 3, 'alpha': 1.0}
            }
            style_ind = {
                'Det3': {'color': '#7fb3d5', 'marker': '.', 'ls': '-', 'label': f'Shifted File | Det 3', 'lw': 1.0, 'alpha': 0.4, 'zorder': 5},
                'Det4': {'color': '#e6b0aa', 'marker': '.', 'ls': '-', 'label': f'Shifted File | Det 4', 'lw': 1.0, 'alpha': 0.4, 'zorder': 5}
            }

            for k in range(max_n):
                ax = axes[k]
                
                for det in ['Det3', 'Det4']:
                    all_V_shifted = []
                    all_P = []
                    added_ind_label = False
                    
                    # A. Plot individuals and gather data
                    for i, sweep in enumerate(sweeps[direction]):
                        if n_sub not in sweep['data'] or det not in sweep['data'][n_sub]:
                            continue
                            
                        V = np.array(sweep['V'])
                        shift = shifts[direction][i]
                        V_shift = V + shift
                        
                        p_list = sweep['data'][n_sub][det]
                        P_k = np.array([p[k] if p is not None and len(p) > k else 0 for p in p_list])
                        
                        if sum(P_k) > 0:
                            lbl = style_ind[det]['label'] if not added_ind_label else ""
                            added_ind_label = True
                            ax.plot(V_shift, P_k, color=style_ind[det]['color'], marker=style_ind[det]['marker'],
                                    markersize=4, linestyle=style_ind[det]['ls'], lw=style_ind[det]['lw'], 
                                    alpha=style_ind[det]['alpha'], zorder=style_ind[det]['zorder'], label=lbl)
                            
                            all_V_shifted.append(V_shift)
                            all_P.append(P_k)
                            
                    # B. Calculate and Plot the Interpolated Average
                    if all_V_shifted:
                        # Find overlapping bounds for the x-axis
                        min_v = max([np.min(v) for v in all_V_shifted])
                        max_v = min([np.max(v) for v in all_V_shifted])
                        
                        # Fallback if overlapping bounds are inverted due to extreme shifts
                        if min_v >= max_v:
                            min_v = min([np.min(v) for v in all_V_shifted])
                            max_v = max([np.max(v) for v in all_V_shifted])
                            
                        # Generate 200 points for smooth averaging
                        common_V = np.linspace(min_v, max_v, 200)
                        interp_arrays = []
                        
                        for V_s, P_k in zip(all_V_shifted, all_P):
                            if len(V_s) > 1:
                                interp_p = np.interp(common_V, V_s, P_k, left=np.nan, right=np.nan)
                                interp_arrays.append(interp_p)
                                
                        if interp_arrays:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                avg_P = np.nanmean(np.array(interp_arrays), axis=0)
                            
                            ax.plot(common_V, avg_P, color=style_avg[det]['color'], marker=style_avg[det]['marker'], 
                                    markersize=style_avg[det]['ms'], markevery=10, linestyle=style_avg[det]['ls'], 
                                    lw=style_avg[det]['lw'], alpha=style_avg[det]['alpha'], 
                                    zorder=style_avg[det]['zorder'], label=style_avg[det]['label'])

                ax.set_title(f"$P(n={k})$", pad=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                if k % cols == 0: ax.set_ylabel("Probability")
                if k >= max_n - cols: ax.set_xlabel("Voltage Shifted (V)")

            for k in range(max_n, len(axes)): fig.delaxes(axes[k])

            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label and label not in labels:
                        handles.append(handle)
                        labels.append(label)
                        
            if handles:
                fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.20, top=0.9)
            plt.show()


# ==========================================
# 4. NEW: TIME-SERIES FRINGE FITTING & DRIFT
# ==========================================
def fringe_model(V, A, omega, phi, y0):
    """Cosine interference fringe model."""
    return A * np.cos(omega * V + phi) + y0

def analyze_fringe_drifts(main_folder, time_step_minutes=14):
    """
    Fits individual time-series sweeps to a cosine fringe model.
    Extracts Visibility, Phase (X-shift), and Offset (Y-shift).
    Plots Raw Fits, Drift over time, and Y-Centered data.
    """
    print(f"Scanning {main_folder} for voltage folders (Fringe Fitting mode)...")
    
    # Structure: sweeps[direction][file_idx] = {'data': { n_sub: [ {'V': v, 'Det3': p3, 'Det4': p4}, ... ] }}
    sweeps = {'UP': [], 'DOWN': []}
    folder_map = {'UP': {}, 'DOWN': {}}

    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    for folder in subfolders:
        match = re.search(r"voltage_([0-9\.]+)V_(UP|DOWN)", folder, re.IGNORECASE)
        if match:
            v = float(match.group(1))
            direction = match.group(2).upper()
            folder_map[direction][v] = folder

    # 1. Parse Data into Time Sweeps
    for direction in ['UP', 'DOWN']:
        voltages = sorted(list(folder_map[direction].keys()))
        if not voltages: continue
        
        # Find maximum number of files (sweeps) recorded in this direction
        max_files = 0
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            max_files = max(max_files, len(txt_files))
            
        print(f"Processing {max_files} sweeps for {direction} direction...")
        
        for i in range(max_files):
            sweeps[direction].append({'data': {}})
            
        for v in voltages:
            folder_path = os.path.join(main_folder, folder_map[direction][v], "data_save3")
            if not os.path.exists(folder_path): continue
            
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            # NATURAL SORT: Ensures file_2.txt comes before file_10.txt to preserve time alignment
            txt_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
            
            for i, file in enumerate(txt_files):
                h_list, m_mat = get_detector2_data_from_file(file)
                if m_mat is None: continue
                
                # Unheralded
                p3, p4 = extract_marginals(m_mat)
                if -1 not in sweeps[direction][i]['data']:
                    sweeps[direction][i]['data'][-1] = []
                
                # Pack safely into a single object
                sweeps[direction][i]['data'][-1].append({'V': v, 'Det3': p3, 'Det4': p4})
                
                # Heralded
                if h_list:
                    for item in h_list:
                        n_sub = item['n_sub']
                        p3, p4 = extract_marginals(item['matrix'])
                        if n_sub not in sweeps[direction][i]['data']:
                            sweeps[direction][i]['data'][n_sub] = []
                            
                        sweeps[direction][i]['data'][n_sub].append({'V': v, 'Det3': p3, 'Det4': p4})

    # 2. Fit Data and Plot
    fit_and_plot_fringes(sweeps, time_step_minutes)

def fit_and_plot_fringes(sweeps, time_step_minutes):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    for direction in ['UP', 'DOWN']:
        if not sweeps[direction]: continue
        
        n_subs = set()
        for sw in sweeps[direction]:
            n_subs.update(sw['data'].keys())
        n_subs = sorted(list(n_subs))

        # NEW: Dictionary to collect Fisher(V) data to plot everything together
        fisher_vs_voltage_data = {}

        for n_sub in n_subs:
            # Determine max_n for grid (Detector 4 Only)
            max_n = 0
            for sw in sweeps[direction]:
                if n_sub in sw['data']:
                    for pt in sw['data'][n_sub]:
                        if pt['Det4'] is not None:
                            max_n = max(max_n, len(pt['Det4']))
                            
            # NEW: Cap the maximum photon number to n=5
            max_n = min(max_n, 6) 
            
            if max_n == 0: continue
            
            # FIT STORAGE: fits[det][n][file_idx] = parameters
            fits = {'Det4': [{} for _ in range(max_n)]}
            raw_data = {'Det4': [{} for _ in range(max_n)]}
            
            num_files = len(sweeps[direction])
            cmap_base = plt.get_cmap('plasma')
            
            # --- EXECUTE ROBUST FITS (Detector 4 Only) ---
            for i, sweep in enumerate(sweeps[direction]):
                if n_sub not in sweep['data']: continue
                
                data_points = sweep['data'][n_sub]
                V_arr_raw = [pt['V'] for pt in data_points]
                
                if len(V_arr_raw) < 4: continue # Need minimum points to fit a curve
                
                sort_idx = np.argsort(V_arr_raw)
                V_arr = np.array(V_arr_raw)[sort_idx]
                time_val = i * time_step_minutes
                
                for det in ['Det4']:
                    p_list = [pt[det] for pt in data_points]
                    
                    # Remember the dominant frequency found in strong signals
                    ref_omega = None 
                    
                    for k in range(max_n):
                        P_k = np.array([p[k] if p is not None and len(p) > k else 0 for p in p_list])[sort_idx]
                        
                        if sum(P_k) < 1e-5: continue # Too weak to fit
                        
                        raw_data[det][k][i] = {'V': V_arr, 'P': P_k}
                        
                        # --- 1. DATA SCALING (For Weighting Only) ---
                        scale_factor = np.max(P_k)
                        P_k_scaled = P_k / scale_factor
                        
                        # --- 2. AGGRESSIVE PEAK WEIGHTING ---
                        peak_weights = 1.0 / (P_k_scaled**3 + 0.01)
                        is_weak = scale_factor < 0.05
                        
                        best_popt = None
                        best_err = np.inf
                        
                        # Determine Physics Input State
                        n_in = 0 if n_sub <= 0 else (1 if n_sub == 1 else None)
                        
                        v_span = max(V_arr[-1] - V_arr[0], 0.1)
                        omega_guesses = [ref_omega] if ref_omega is not None else [(2 * np.pi / v_span) * m for m in [2.0, 1.0, 3.0, 0.5, 4.0]]
                        
                        if n_in is not None:
                            # --- SU(1,1) ANALYTICAL PHYSICS FIT ---
                            su11_model = generate_su11_analytic_model(n_detect=k, n_in=n_in)
                            
                            for omg in omega_guesses:
                                for phi_guess in [0, np.pi/2, np.pi, 3*np.pi/2]:
                                    try:
                                        popt, _ = curve_fit(su11_model, V_arr, P_k, 
                                                            p0=[0.5, 0.5, 0.5, omg, phi_guess], 
                                                            bounds=([0.0, 0.0, 0.0, 0.0, -np.inf], [2.0, 2.0, 1.0, np.inf, np.inf]),
                                                            sigma=peak_weights, maxfev=10000)
                                        fit_P = su11_model(V_arr, *popt)
                                        err = np.sum((P_k - fit_P)**2)
                                        if err < best_err:
                                            best_err = err
                                            best_popt = {'type': 'su11', 'r1': popt[0], 'r2': popt[1], 'eta': popt[2], 'omega': popt[3], 'phi': popt[4], 'n_in': n_in}
                                    except Exception:
                                        continue
                        else:
                            # --- FALLBACK COSINE FIT (For N1 >= 2) ---
                            y0_guess = np.mean(P_k_scaled)
                            A_guess = (np.max(P_k_scaled) - np.min(P_k_scaled)) / 2
                            vis_guess = min(A_guess / y0_guess if y0_guess > 0 else 0.0, 1.0)
                            
                            def bounded_model(V, vis, omega, phi, y0_norm):
                                return y0_norm * (1 + vis * np.cos(omega * V + phi))
                                
                            for omg in omega_guesses:
                                for phi_guess in [0, np.pi/2, np.pi, 3*np.pi/2]:
                                    try:
                                        popt, _ = curve_fit(bounded_model, V_arr, P_k_scaled, 
                                                            p0=[vis_guess, omg, phi_guess, y0_guess], 
                                                            bounds=([0.0, 0.0, -np.inf, 0.0], [1.0, np.inf, np.inf, 2.0]),
                                                            sigma=peak_weights, maxfev=10000)
                                        y0_fit = popt[3] * scale_factor
                                        A_fit = popt[0] * y0_fit
                                        fit_P = y0_fit * (1 + popt[0] * np.cos(popt[1] * V_arr + popt[2]))
                                        err = np.sum((P_k - fit_P)**2)
                                        
                                        if err < best_err:
                                            best_err = err
                                            best_popt = {'type': 'cos', 'A': A_fit, 'omega': popt[1], 'phi': popt[2], 'y0': y0_fit, 'vis': popt[0]}
                                    except Exception:
                                        continue

                        # --- STORE BEST FIT PARAMETERS ---
                        if best_popt is not None:
                            best_popt['time'] = time_val
                            
                            # Wrap Phase
                            phi_raw = best_popt['phi']
                            best_popt['phi'] = (phi_raw + np.pi) % (2 * np.pi) - np.pi
                            
                            if best_popt['type'] == 'su11':
                                # Compute effective Vis and y0 for the Drift Plots
                                model_func = generate_su11_analytic_model(k, best_popt['n_in'])
                                dense_V = np.linspace(0, 2*np.pi/best_popt['omega'], 100)
                                dense_P = model_func(dense_V, best_popt['r1'], best_popt['r2'], best_popt['eta'], best_popt['omega'], best_popt['phi'])
                                max_P, min_P = np.max(dense_P), np.min(dense_P)
                                best_popt['y0'] = (max_P + min_P) / 2
                                best_popt['vis'] = (max_P - min_P) / (max_P + min_P) if (max_P + min_P) > 0 else 0
                            else:
                                # Standardize generic cosine A sign
                                if best_popt['A'] < 0:
                                    best_popt['A'] = -best_popt['A']
                                    best_popt['phi'] = (best_popt['phi'] + np.pi) % (2 * np.pi) - np.pi
                            
                            fits[det][k][i] = best_popt
                            
                            if ref_omega is None and not is_weak:
                                ref_omega = best_popt['omega']
                        else:
                            print(f"Fit completely failed for {direction} | {det} | P({k}) at file {i}")
            
            # --- INSERT FISHER ANALYSIS HERE ---
            # Now that all fits are stored in the 'fits' and 'raw_data' dictionaries:
            metrics = compute_raw_fisher_sectors(raw_data, sectors=[1, 2, 3, 4, 5])
            plot_fisher_comparison_summed(metrics)
            
            # NEW: Compute Fisher over Voltage for the first file (file_idx=0)
            f_v_data = compute_fisher_over_voltage(raw_data, fits, file_idx=0)
            if f_v_data is not None:
                fisher_vs_voltage_data[n_sub] = f_v_data

            # --- PLOT 1: RAW DATA + FITS ---
            cols = min(3, max_n)
            rows = math.ceil(max_n / cols)
            fig1, axes1 = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2), sharex=True)
            if max_n == 1: axes1 = [axes1]
            else: axes1 = axes1.flatten()
            
            title_base = "Unheralded" if n_sub == -1 else f"Heralded $N_1 = {n_sub}$"
            fig1.suptitle(f"Fringe Fits [{direction}]: {title_base}\n(Detector 4 Only)", fontsize=16, fontweight='bold')
            
            for k in range(max_n):
                ax = axes1[k]
                for det, m_fmt in [('Det4', 'x')]:
                    for i in range(num_files):
                        if i in raw_data[det][k]:
                            V = raw_data[det][k][i]['V']
                            P = raw_data[det][k][i]['P']
                            c = cmap_base(i / max(1, num_files - 1))
                            
                            ax.scatter(V, P, color=c, marker=m_fmt, s=15, alpha=0.6)
                            
                            if i in fits[det][k]:
                                fit = fits[det][k][i]
                                V_dense = np.linspace(V[0], V[-1], 100)
                                
                                # Evaluate the correct model type
                                if fit['type'] == 'su11':
                                    model_func = generate_su11_analytic_model(k, fit['n_in'])
                                    P_fit = model_func(V_dense, fit['r1'], fit['r2'], fit['eta'], fit['omega'], fit['phi'])
                                else:
                                    P_fit = fringe_model(V_dense, fit['A'], fit['omega'], fit['phi'], fit['y0'])
                                    
                                ax.plot(V_dense, P_fit, color=c, linewidth=1.5, alpha=0.8)
                
                ax.set_title(f"$P(n={k})$")
                ax.grid(True, ls='--', alpha=0.5)
                if k >= max_n - cols: ax.set_xlabel("Voltage (V)")
                
            for k in range(max_n, len(axes1)): fig1.delaxes(axes1[k])
            plt.tight_layout()
            plt.show()

            # --- PLOT 2: DRIFTS OVER TIME ---
            has_fits = any(fits['Det4'][k] for k in range(max_n))
            
            if has_fits:
                fig2, axes2 = plt.subplots(max_n, 3, figsize=(15, 3.5 * max_n))
                if max_n == 1: axes2 = np.array([axes2])
                fig2.suptitle(f"Parameter Drifts [{direction}]: {title_base}\n(Detector 4 Only)", fontsize=16, fontweight='bold')
                
                for k in range(max_n):
                    ax_vis = axes2[k, 0]
                    ax_phi = axes2[k, 1]
                    ax_y0  = axes2[k, 2]
                    
                    for det, style in [('Det4', {'c': '#c0392b', 'm': 's'})]:
                        times = [fits[det][k][i]['time'] for i in sorted(fits[det][k].keys())]
                        if not times: continue
                        
                        vis = [fits[det][k][i]['vis'] for i in sorted(fits[det][k].keys())]
                        phi_raw = [fits[det][k][i]['phi'] for i in sorted(fits[det][k].keys())]
                        y0  = [fits[det][k][i]['y0']  for i in sorted(fits[det][k].keys())]
                        
                        # Unwrap to prevent artificial 2*pi jumps, then make relative to t=0
                        phi_unwrapped = np.unwrap(phi_raw)
                        phi_relative = phi_unwrapped - phi_unwrapped[0]
                        
                        ax_vis.plot(times, vis, marker=style['m'], color=style['c'], ls='-', lw=2, label=det)
                        ax_phi.plot(times, phi_relative, marker=style['m'], color=style['c'], ls='-', lw=2)
                        ax_y0.plot(times, y0, marker=style['m'], color=style['c'], ls='-', lw=2)
                    
                    ax_vis.set_ylabel(f"P({k}) Visibility")
                    ax_phi.set_ylabel(f"P({k}) Relative Phase $\Delta\phi$ (rad)")
                    ax_y0.set_ylabel(f"P({k}) Y-Offset $y_0$")
                    
                    for ax in [ax_vis, ax_phi, ax_y0]:
                        ax.grid(True, ls='--', alpha=0.5)
                        if k == max_n - 1: ax.set_xlabel("Time (minutes)")
                        
                    if k == 0: ax_vis.legend(loc='best')
                        
                plt.tight_layout()
                plt.show()

                # --- PLOT 3: X-ALIGNED DATA (Phase Compensated) ---
                fig3, axes3 = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows + 2), sharex=True)
                if max_n == 1: axes3 = [axes3]
                else: axes3 = axes3.flatten()
                fig3.suptitle(f"Phase-Aligned Fringes [{direction}]: {title_base}\n(Detector 4 Only, X-Shifted using $P(n=0)$ phase)", fontsize=16, fontweight='bold')
                
                # --- PRE-CALCULATE SHIFT USING ONLY P(n=0) ---
                file_shifts = {}
                ref_k = 0 # Force reference to P(n=0)
                first_valid_i_0 = next((i for i in range(num_files) if i in fits['Det4'][ref_k]), None) if max_n > 0 else None
                ref_phi_0 = fits['Det4'][ref_k][first_valid_i_0]['phi'] if first_valid_i_0 is not None else 0
                
                for i in range(num_files):
                    if first_valid_i_0 is not None and i in fits['Det4'][ref_k]:
                        fit_0 = fits['Det4'][ref_k][i]
                        delta_phi_0 = fit_0['phi'] - ref_phi_0
                        file_shifts[i] = delta_phi_0 / fit_0['omega']
                    else:
                        file_shifts[i] = 0.0 # Fallback to no shift if P(0) failed to fit

                for k in range(max_n):
                    ax = axes3[k]
                    
                    for det, m_fmt in [('Det4', 'x')]:
                        for i in range(num_files):
                            if i in raw_data[det][k] and i in fits[det][k]:
                                V = raw_data[det][k][i]['V']
                                P = raw_data[det][k][i]['P']
                                fit = fits[det][k][i]
                                
                                c = cmap_base(i / max(1, num_files - 1))
                                
                                # Shift the voltage axis using the global file shift from P(n=0)
                                delta_V = file_shifts[i]
                                
                                V_shifted = V + delta_V
                                ax.scatter(V_shifted, P, color=c, marker=m_fmt, s=15, alpha=0.6)
                                
                                # Align the fit curve to the new grid
                                V_dense = np.linspace(V[0], V[-1], 100)
                                
                                # Evaluate the correct model type
                                if fit['type'] == 'su11':
                                    model_func = generate_su11_analytic_model(k, fit['n_in'])
                                    P_fit = model_func(V_dense, fit['r1'], fit['r2'], fit['eta'], fit['omega'], fit['phi'])
                                else:
                                    P_fit = fringe_model(V_dense, fit['A'], fit['omega'], fit['phi'], fit['y0'])
                                    
                                V_dense_shifted = V_dense + delta_V
                                ax.plot(V_dense_shifted, P_fit, color=c, linewidth=1.5, alpha=0.8)
                    
                    ax.set_title(f"$P(n={k})_{{aligned}}$")
                    ax.grid(True, ls='--', alpha=0.5)
                    if k >= max_n - cols: ax.set_xlabel("Aligned Voltage (V)")
                    
                for k in range(max_n, len(axes3)): fig3.delaxes(axes3[k])
                plt.tight_layout()
                plt.show()

        # --- PLOT 4: COMBINED FISHER VS VOLTAGE ---
        # Plot the collected unheralded and heralded data together after the n_sub loop
        if fisher_vs_voltage_data:
            plot_fisher_vs_voltage(fisher_vs_voltage_data, direction)



def compute_phase_metrics(fits, raw_data, n_subs):
    """
    Computes Fisher Information, Phase Sensitivity, and SQL comparison.
    
    Returns: 
        Dict[det][n_sub][file_idx] = {'fisher': F, 'sensitivity': dPhi, 'sql': sql, 'time': time}
    """
    phase_metrics = {}

    for det in fits.keys():
        phase_metrics[det] = {}
        for k in range(len(fits[det])):
            phase_metrics[det][k] = {}
            for i, fit in fits[det][k].items():
                if i not in raw_data[det][k]: continue
                
                A, phi, omega, y0 = fit['A'], fit['phi'], fit['omega'], fit['y0']
                V = raw_data[det][k][i]['V']
                P_arr = raw_data[det][k][i]['P']
                
                # 1. Fisher Information (Classical)
                # Model P(V) = A*cos(omega*V + phi) + y0
                # dP/d_phi = -A*sin(omega*V + phi)
                fringe_phase = omega * V + phi
                derivative = -A * np.sin(fringe_phase)
                P_v = A * np.cos(fringe_phase) + y0
                
                # Sum over all V points (Fisher info is additive for independent measurements)
                fisher_info = np.sum((derivative**2) / (P_v + 1e-9))
                phase_sensitivity = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else np.inf
                
                # 2. SQL Calculation
                # SQL = 1 / sqrt(<n>)
                mean_n = np.sum(P_arr * np.arange(len(P_arr)))
                sql = 1.0 / np.sqrt(mean_n) if mean_n > 0 else np.inf
                
                phase_metrics[det][k][i] = {
                    'fisher': fisher_info,
                    'sensitivity': phase_sensitivity,
                    'sql': sql,
                    'time': fit['time']
                }
    return phase_metrics

def plot_fisher_comparison(phase_metrics, direction):
    """
    Plots Fisher Information and Sensitivity side-by-side for all n_subs.
    """
    # Use a colormap to distinguish different n_sub states
    cmap = plt.get_cmap('viridis')
    n_keys = sorted(phase_metrics['Det4'].keys())
    
    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, n in enumerate(n_keys):
        color = cmap(idx / max(1, len(n_keys) - 1))
        label = "Unheralded" if n == -1 else f"$N_1 = {n}$"
        
        # Sort by time
        data = phase_metrics['Det4'][n]
        times = [d['time'] for i, d in sorted(data.items())]
        fisher = [d['fisher'] for i, d in sorted(data.items())]
        sens = [d['sensitivity'] for i, d in sorted(data.items())]
        sql = [d['sql'] for i, d in sorted(data.items())]
        
        # Plot Fisher
        ax_f.plot(times, fisher, marker='o', color=color, label=label, alpha=0.7)
        
        # Plot Sensitivity
        ax_s.plot(times, sens, marker='o', color=color, label=label, alpha=0.7)
        # Only plot SQL once (it's similar for all unless heralded strongly modifies mean n)
        if idx == 0:
            ax_s.plot(times, sql, linestyle='--', color='black', alpha=0.5, label='Classical Limit (SQL)')

    # Formatting Fisher Plot
    ax_f.set_title("Fisher Information $F_\phi$")
    ax_f.set_xlabel("Time (min)")
    ax_f.set_ylabel("Fisher Information")
    ax_f.grid(True, linestyle=':', alpha=0.6)
    
    # Formatting Sensitivity Plot
    ax_s.set_title("Phase Sensitivity $\Delta \phi$")
    ax_s.set_xlabel("Time (min)")
    ax_s.set_ylabel("Sensitivity (rad)")
    ax_s.set_yscale('log')
    ax_s.grid(True, linestyle=':', alpha=0.6)
    ax_s.legend()
    
    fig.suptitle(f"Quantum Phase Analysis [{direction}]", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compute_raw_fisher(raw_data, n_subs):
    """
    Computes Fisher Information using RAW probability points rather than fits.
    This captures the real detector noise.
    
    Formula: F = sum( (dP/dphi)^2 / P )
    """
    phase_metrics = {}

    for det in raw_data.keys():
        phase_metrics[det] = {}
        for n_sub in raw_data[det].keys():
            phase_metrics[det][n_sub] = {}
            for i in raw_data[det][n_sub].keys():
                data = raw_data[det][n_sub][i]
                V, P = data['V'], data['P']
                
                # To get dP/dphi from raw data, we use the gradient of the points
                # We assume the phase modulation is linear across the voltage scan
                dP_dV = np.gradient(P, V)
                
                # We need dP/dphi. Since V = phase / omega, dphi = omega * dV
                # dP/dphi = (dP/dV) / omega
                # We estimate omega from the fit frequency (assuming consistent omega)
                omega = 1.0 # Or use your fitted omega here
                
                dP_dphi = dP_dV / omega
                
                # Fisher = sum( (dP/dphi)^2 / P )
                # Add tiny epsilon to P to avoid div by zero
                fisher = np.sum((dP_dphi**2) / (P + 1e-9))
                sensitivity = 1.0 / np.sqrt(fisher) if fisher > 0 else np.inf
                
                phase_metrics[det][n_sub][i] = {'fisher': fisher, 'sensitivity': sensitivity}
                
    return phase_metrics

def compute_raw_fisher_sectors(raw_data, sectors=[1, 2, 3, 4, 5]):
    """
    Computes Raw Fisher Information for individual photon sectors (n=1 to 5)
    and returns the Total Summed Fisher Information.
    
    raw_data expected structure: raw_data[det][k][i] = {'V': V_arr, 'P': P_arr}
    """
    sector_fishers = {}
    
    for n in sectors:
        det = 'Det4' # Focusing on Detector 4
        if det not in raw_data:
            continue
            
        # raw_data[det] is a list. Ensure n is a valid index!
        if n >= len(raw_data[det]):
            continue
            
        sweep_dict = raw_data[det][n]
        if not sweep_dict:
            continue
            
        sector_fishers[n] = {'fisher': [], 'sweep_idx': []}
        
        # Process each sweep (i) in order
        for i in sorted(sweep_dict.keys()):
            data = sweep_dict[i]
            V = np.array(data['V'])
            P = np.array(data['P'])
            
            if len(P) < 2: 
                continue
            
            # Raw numerical derivative dP/dV
            dP_dV = np.gradient(P, V)
            
            # Fisher Information wrt Voltage
            fisher = np.sum((dP_dV**2) / (P + 1e-9))
            
            sector_fishers[n]['fisher'].append(fisher)
            sector_fishers[n]['sweep_idx'].append(i) # X-axis is time/sweep index

    return sector_fishers

def plot_fisher_comparison_summed(sector_metrics):
    """
    Plots the Fisher Information for the requested sectors and the Total State.
    """
    if not sector_metrics:
        print("No Fisher metrics found to plot.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_fishers = []
    common_idx = []
    
    # Plot individual sectors and accumulate total
    for n in sorted(sector_metrics.keys()):
        if not sector_metrics[n]['fisher']:
            continue
            
        ax.plot(sector_metrics[n]['sweep_idx'], sector_metrics[n]['fisher'], 
                'o-', alpha=0.6, label=f'Sector $P(n={n})$')
        
        # Accumulate for total (safely aligning arrays)
        if len(total_fishers) == 0:
            total_fishers = np.array(sector_metrics[n]['fisher'])
            common_idx = sector_metrics[n]['sweep_idx']
        else:
            min_len = min(len(total_fishers), len(sector_metrics[n]['fisher']))
            total_fishers = total_fishers[:min_len] + np.array(sector_metrics[n]['fisher'][:min_len])
            common_idx = common_idx[:min_len]
    
    # Plot Total Summed Fisher
    if len(total_fishers) > 0:
        ax.plot(common_idx, total_fishers, 'k-', lw=3, label='Total State (Summed)')
    
    ax.set_title("Summed Fisher Information: Multi-Photon Contribution over Time")
    ax.set_ylabel("Raw Fisher Information ($F_V$)")
    ax.set_xlabel("Sweep / File Index (Time)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def compute_fisher_over_voltage(raw_data, fits, file_idx=0):
    """
    Computes the Classical Fisher Information over Voltage (Phase) 
    for a specific sweep file, aggregating over all photon sectors k.
    Uses a hybrid approach: Analytical dP/dphi from fits, but raw P(V) for noise.
    """
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    
    det = 'Det4'
    if det not in raw_data:
        return None
        
    # Find valid k sectors that contain the target file_idx
    valid_ks = [k for k in range(len(raw_data[det])) if file_idx in raw_data[det][k]]
    if not valid_ks:
        # Fallback: try to find ANY file index if 0 is missing
        all_files = []
        for k in range(len(raw_data[det])):
            all_files.extend(list(raw_data[det][k].keys()))
        if not all_files:
            return None
        file_idx = min(all_files)
        valid_ks = [k for k in range(len(raw_data[det])) if file_idx in raw_data[det][k]]

    if not valid_ks:
        return None

    # Use the voltage array from the first valid sector
    V_arr = raw_data[det][valid_ks[0]][file_idx]['V']
    F_phi_total = np.zeros_like(V_arr)
    mean_n = np.zeros_like(V_arr)
        
    for k in valid_ks:
        data = raw_data[det][k][file_idx]
        P = data['P']
        
        if len(P) != len(V_arr):
            continue
            
        # Smooth the probabilities slightly for the denominator (noise floor)
        P_smooth = gaussian_filter1d(P, sigma=1.5)
        
        # --- HYBRID APPROACH: Analytical Derivative from Fits ---
        if file_idx in fits[det][k] and 'A' in fits[det][k][file_idx]:
            fit = fits[det][k][file_idx]
            A = fit['A']
            omg = fit['omega']
            phi = fit['phi']
            
            # Analytical derivative w.r.t phase: dP/d(phi) = -A * sin(omega*V + phi)
            dP_dphi = -A * np.sin(omg * V_arr + phi)
        else:
            # Fallback to 0 if the curve fitter completely failed for this specific n-photon sector
            dP_dphi = np.zeros_like(V_arr)
        
        # Fisher Information wrt Phase for this sector
        F_k = (dP_dphi**2) / (P_smooth + 1e-9)
        
        F_k = np.nan_to_num(F_k, nan=0.0, posinf=0.0, neginf=0.0) 
        F_phi_total += F_k
        
        # Accumulate mean photon number for the Standard Noise Limit (SNL)
        mean_n += k * P_smooth
        
    # The true Classical Limit (SNL) for the input state is constant. 
    # Since photons oscillate between Det 3 and 4, the max of Det 4 represents the total input.
    constant_snl = np.max(mean_n) * np.ones_like(V_arr)
    
    return {'V': V_arr, 'F_phi': F_phi_total, 'SNL': constant_snl, 'file_idx': file_idx}

def plot_fisher_vs_voltage(fisher_data, direction):
    """
    Plots Fisher Information vs Voltage for unheralded and heralded states
    on a single graph, comparing them to their respective SNLs.
    """
    
    if not fisher_data:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Distinct colors for different states
    colors = {-1: '#7f8c8d', 0: '#2980b9', 1: '#c0392b', 2: '#27ae60', 3: '#8e44ad', 4: '#e67e22'}
    
    file_idx = None
    for n_sub in sorted(fisher_data.keys()):
        if n_sub > 2:
            continue # Exclude higher N_1 states to keep lower features clearly visible
            
        data = fisher_data[n_sub]
        V = data['V']
        F = data['F_phi']
        SNL = data['SNL']
        file_idx = data['file_idx']
        
        label = "Unheralded" if n_sub == -1 else f"Heralded $N_1={n_sub}$"
        c = colors.get(n_sub, '#2c3e50')
        
        # Plot the Experimental Fisher Information
        ax.plot(V, F, '-', color=c, lw=2.5, label=f"{label} ($F_\\phi$)")
        
        # Plot the Shot Noise Limit (SNL)
        ax.plot(V, SNL, '--', color=c, lw=1.5, alpha=0.8)
        
    ax.set_title(f"Phase Sensitivity ($F_\\phi$) vs. Piezo Voltage\n[File {file_idx}, {direction} Sweep, Det 4]")
    ax.set_xlabel("Piezo Voltage (V)")
    ax.set_ylabel("Classical Fisher Information ($F_\\phi$)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(0,1)
    
    # Limit the y-axis dynamically to avoid huge noise spikes warping the graph
    # Filter out NaNs/Infs and EXCLUDE n_sub > 2 so the scale perfectly matches the visible data
    all_F = [val for n_sub, d in fisher_data.items() if n_sub <= 2 for val in d['F_phi'] if np.isfinite(val)]
    if all_F:
        max_y = np.percentile(all_F, 98) * 1.5 
        if max_y <= 0 or np.isnan(max_y): 
            max_y = 1.0 # Safe fallback
    ax.set_ylim(0, 0.4)
        
    # Custom Legend handles to explain the lines
    handles, labels = ax.get_legend_handles_labels()
    snl_line = mlines.Line2D([], [], color='black', linestyle='--', lw=1.5, alpha=0.8, label='Standard Noise Limit (SNL = $\\langle n \\rangle$)')
    handles.append(snl_line)
    labels.append('Standard Noise Limit (SNL = $\\langle n \\rangle$)')
    
    ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()

def plot_phase_sensitivity_vs_voltage(fisher_data, direction):
    """
    Plots Experimental Phase Sensitivity (Delta phi) using the Cramer-Rao Bound.
    Delta phi = 1 / sqrt(F_phi).
    Lower is better (higher resolution).
    """
    if not fisher_data:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {-1: '#7f8c8d', 0: '#2980b9', 1: '#c0392b', 2: '#27ae60', 3: '#8e44ad', 4: '#e67e22'}
    
    file_idx = None
    all_delta_phi = []
    
    for n_sub in sorted(fisher_data.keys()):
        if n_sub > 2:
            continue # Exclude noisy higher N_1 states
            
        data = fisher_data[n_sub]
        V = data['V']
        F = data['F_phi']
        SNL = data['SNL']
        file_idx = data['file_idx']
        
        # Cramer-Rao Bound: Delta Phi = 1 / sqrt(F)
        # We add a tiny number to the denominator to prevent division by zero in the dark fringes
        delta_phi = 1.0 / np.sqrt(F + 1e-6)
        delta_phi_sql = 1.0 / np.sqrt(SNL + 1e-6)
        
        # Clip absurdly high values (which just mean "no sensitivity here") for clean plotting
        delta_phi = np.clip(delta_phi, 0, 10) 
        
        label = "Unheralded" if n_sub == -1 else f"Heralded $N_1={n_sub}$"
        c = colors.get(n_sub, '#2c3e50')
        
        ax.plot(V, delta_phi, '-', color=c, lw=2.5, label=f"{label} ($\\Delta\\phi$)")
        ax.plot(V, delta_phi_sql, '--', color=c, lw=1.5, alpha=0.8)
        
        # Collect valid valleys (highest sensitivity / lowest delta_phi) to set y-axis bounds
        valid_valleys = [v for v in delta_phi if v < 5.0]
        all_delta_phi.extend(valid_valleys)
        
    ax.set_title(f"Experimental Phase Sensitivity ($\\Delta\\phi$) vs. Piezo Voltage\n[File {file_idx}, {direction} Sweep, Det 4]")
    ax.set_xlabel("Piezo Voltage (V)")
    ax.set_ylabel("Phase Uncertainty $\\Delta\\phi$ (radians)\n[Lower is Better]")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set Y-axis so the best sensitivities (the valleys) are clearly visible
    if all_delta_phi:
        min_y = max(0, np.min(all_delta_phi) * 0.8)
        # Cap the top of the graph so the giant spikes at the dark fringes don't ruin the view
        max_y = np.percentile(all_delta_phi, 85) * 1.5
        ax.set_ylim(min_y, max_y)
        
    handles, labels = ax.get_legend_handles_labels()
    snl_line = mlines.Line2D([], [], color='black', linestyle='--', lw=1.5, alpha=0.8, label='SQL ($\\Delta\\phi = 1/\\sqrt{\\langle n \\rangle}$)')
    handles.append(snl_line)
    labels.append('SQL ($\\Delta\\phi = 1/\\sqrt{\\langle n \\rangle}$)')
    
    ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

def generate_su11_analytic_model(n_detect, n_in, m_max=15):
    """Generates the exact analytical P(n) probability curve for an SU(1,1) interferometer."""
    def su11_model(V, r1, r2, eta, omega, phi0):
        phi = omega * V + phi0
        # Protect against negative arguments due to float precision limits
        cosh_reff = np.sqrt(np.maximum(1.0, np.cosh(r1 - r2)**2 + np.sinh(2*r1)*np.sinh(2*r2)*np.sin(phi/2)**2))
        tanh_reff = np.sqrt(np.maximum(0.0, 1.0 - 1.0/(cosh_reff**2)))
        
        P_ideal = []
        for m in range(m_max):
            if n_in == 0:
                if m % 2 != 0: 
                    P_ideal.append(np.zeros_like(V))
                else:
                    k_idx = m // 2
                    coeff = math.factorial(2*k_idx) / ((2**(2*k_idx)) * (math.factorial(k_idx)**2))
                    P_ideal.append((1.0 / cosh_reff) * coeff * (tanh_reff**(2*k_idx)))
            elif n_in == 1:
                if m % 2 == 0: 
                    P_ideal.append(np.zeros_like(V))
                else:
                    k_idx = (m - 1) // 2
                    coeff = math.factorial(2*k_idx + 1) / ((2**(2*k_idx)) * (math.factorial(k_idx)**2))
                    P_ideal.append((1.0 / (cosh_reff**3)) * coeff * (tanh_reff**(2*k_idx)))
            else:
                P_ideal.append(np.zeros_like(V))
                
        # Loss Mixing
        P_out = np.zeros_like(V)
        for m in range(n_detect, m_max):
            P_out += math.comb(m, n_detect) * (eta**n_detect) * ((1 - eta)**(m - n_detect)) * P_ideal[m]
            
        return P_out
    return su11_model

    # %%
# Example:
#main_folder = r"C:\Users\joh90929\OneDrive - Friedrich-Schiller-Universität Jena\lab_data\new_pnr_data\Data\14_05_2026\piezoscan\HOM_4,4mW_10s\angle_25deg"

main_folder = r"C:\Users\joh90929\OneDrive - Friedrich-Schiller-Universität Jena\lab_data\new_pnr_data\Data\15_05_2026\piezo_scan\sm_and_tm_subtr_4,4mW_10s\HWP2_25deg_HWP4_46deg"    
#analyze_voltage_scan(main_folder)

analyze_voltage_scan_all(main_folder)
#analyze_voltage_scan_coincidences(main_folder)
analyze_fringe_drifts(main_folder, time_step_minutes=14)
#analyze_voltage_scan_coincidences(main_folder)
# %%
