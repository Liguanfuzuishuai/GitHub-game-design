import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import product
import re

def parse_table(text):
    """
    解析用户粘贴的Excel区块数据：
    - 首行为表头（Lat\Lon, lon1, lon2, ...）
    - 首列为纬度
    - 其它为数值，支持NAN, #VALUE!等
    返回 lat_vals, lon_vals, data  (lat, lon均为1维, data为2维)
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    header = re.split(r'\t|,| +', lines[0])[1:]  # 跳过第一个表头单元格
    lon_vals = [float(x) for x in header]
    lat_vals, data = [], []
    for line in lines[1:]:
        parts = re.split(r'\t|,| +', line)
        lat_vals.append(float(parts[0]))
        row = []
        for val in parts[1:]:
            if val in ('NAN', 'NaN', '#VALUE!', ''):
                row.append(np.nan)
            else:
                try:
                    row.append(float(val))
                except:
                    row.append(np.nan)
        data.append(row)
    return np.array(lat_vals), np.array(lon_vals), np.array(data)

def poly_features3(X1, X2, X3, degree):
    terms = []
    powers = []
    for d in range(degree+1):
        for px in range(d+1):
            for py in range(d-px+1):
                pz = d - px - py
                powers.append((px, py, pz))
    for px, py, pz in powers:
        terms.append((X1**px) * (X2**py) * (X3**pz))
    features = np.stack(terms, axis=1)
    return features, powers

def correlation_coefficient(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    numerator = np.sum((y_true - y_true_mean)*(y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean)**2) * np.sum((y_pred - y_pred_mean)**2))
    return numerator / denominator

def fit_and_show(ndvi_text, lst_text, output_widget):
    try:
        lat_vals, lon_vals, ndvi = parse_table(ndvi_text)
        lat_vals2, lon_vals2, lst = parse_table(lst_text)
        if not (np.allclose(lat_vals, lat_vals2) and np.allclose(lon_vals, lon_vals2)):
            messagebox.showerror("Error", "NDVI和LST的经纬度范围不一致！")
            return
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        ndvi_flat = ndvi.flatten()
        lst_flat = lst.flatten()
        mask = ~np.isnan(ndvi_flat) & ~np.isnan(lst_flat)
        lat_flat = lat_flat[mask]
        lon_flat = lon_flat[mask]
        ndvi_flat = ndvi_flat[mask]
        lst_flat = lst_flat[mask]

        # 自动寻找最佳R
        best_r = None
        best_corr = None
        best_coeffs = None
        best_powers = None
        r_list = []
        corr_list = []
        max_poly_degree = 10
        output_lines = []
        for R in range(1, max_poly_degree+1):
            X, powers = poly_features3(lat_flat, lon_flat, ndvi_flat, R)
            coeffs = np.linalg.lstsq(X, lst_flat, rcond=None)[0]
            y_pred = X @ coeffs
            corr = correlation_coefficient(lst_flat, y_pred)
            output_lines.append(f"R={R} 的相关系数 r={corr:.4f}")
            r_list.append(R)
            corr_list.append(corr)
            if (best_corr is None) or (abs(corr) > abs(best_corr)):
                best_corr = corr
                best_r = R
                best_coeffs = coeffs
                best_powers = powers
        output_lines.append(f"\n最佳R={best_r}, 相关系数 r={best_corr:.4f}")
        # 拟合公式
        varnames = ['Lat', 'Lon', 'NDVI']
        expr_terms = []
        for c, (px, py, pz) in zip(best_coeffs, best_powers):
            term = ""
            for v, p in zip(varnames, (px, py, pz)):
                if p == 0:
                    continue
                elif p == 1:
                    term += f"*{v}"
                else:
                    term += f"*{v}^{p}"
            expr_terms.append(f"{c:.6f}{term}" if term else f"{c:.6f}")
        poly_expr = "LST = " + " + ".join(expr_terms)
        output_lines.append("\n最佳多项式拟合代数式为：")
        output_lines.append(poly_expr)

        output_widget.config(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, "\n".join(output_lines))
        output_widget.config(state='disabled')

        # 绘制三维图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lat_flat, lon_flat, lst_flat, c='b', marker='o', label='Real LST')

        LAT_SURF, LON_SURF = np.meshgrid(lat_vals, lon_vals, indexing='ij')
        NDVI_SURF = ndvi
        lat_s = LAT_SURF.flatten()
        lon_s = LON_SURF.flatten()
        ndvi_s = NDVI_SURF.flatten()
        mask_s = ~np.isnan(ndvi_s)
        lat_s = lat_s[mask_s]
        lon_s = lon_s[mask_s]
        ndvi_s = ndvi_s[mask_s]
        X_surf, _ = poly_features3(lat_s, lon_s, ndvi_s, best_r)
        lst_surf = X_surf @ best_coeffs
        LST_SURF = np.full(LAT_SURF.shape, np.nan)
        LST_SURF.flat[mask_s] = lst_surf
        from matplotlib import cm
        surf = ax.plot_surface(LAT_SURF, LON_SURF, LST_SURF, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
        ax.set_xticks(lat_vals)
        ax.set_yticks(lon_vals)
        ax.set_xlabel('Lat')
        ax.set_ylabel('Lon')
        ax.set_zlabel('LST')
        ax.set_title(f'Ternary polynomial fitting (R={best_r}, r={best_corr:.4f})')
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("错误", f"数据解析或拟合出错:\n{e}")

def main():
    root = tk.Tk()
    root.title("Nina's proprietary multivariate polynomial fitting tool")
    root.geometry('1200x800')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=1)

    # NDVI区域
    lbl_ndvi = ttk.Label(frm, text='请粘贴NDVI表格数据（含Lat/Lon表头）:')
    lbl_ndvi.grid(row=0, column=0, sticky='w')
    txt_ndvi = scrolledtext.ScrolledText(frm, width=60, height=16)
    txt_ndvi.grid(row=1, column=0, padx=5, pady=5)

    # LST区域
    lbl_lst = ttk.Label(frm, text='请粘贴LST表格数据（含Lat/Lon表头）:')
    lbl_lst.grid(row=0, column=1, sticky='w')
    txt_lst = scrolledtext.ScrolledText(frm, width=60, height=16)
    txt_lst.grid(row=1, column=1, padx=5, pady=5)

    # 输出区域
    lbl_out = ttk.Label(frm, text='输出结果:')
    lbl_out.grid(row=2, column=0, sticky='w', pady=(10,0))
    txt_out = scrolledtext.ScrolledText(frm, width=120, height=14, font=('Consolas', 10))
    txt_out.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    txt_out.config(state='disabled')

    # 按钮
    btn_fit = ttk.Button(frm, text="FIT", command=lambda: fit_and_show(txt_ndvi.get("1.0", tk.END),
                                                                      txt_lst.get("1.0", tk.END),
                                                                      txt_out))
    btn_fit.grid(row=4, column=0, columnspan=2, pady=15)

    root.mainloop()

if __name__ == '__main__':
    main()