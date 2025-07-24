import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as mcolors # Import untuk BoundaryNorm
from utils import get_db_connection # Asumsi utils.py sudah ada dan berfungsi
from flask import Blueprint, request, jsonify, send_from_directory

heatmap_consumptions_bp = Blueprint('heatmap_consumptions', __name__)
@heatmap_consumptions_bp.route('/heatmap-consumptions', methods=['GET'])
def generate_consumption_heatmap(output_filename="consumption_heatmap.png"):
    """
    Mengambil data konsumsi bulanan dari database, memprosesnya,
    dan menghasilkan heatmap total konsumsi per bulan dan tahun.

    Args:
        output_filename (str): Nama file untuk menyimpan gambar heatmap.
            Akan disimpan di direktori static/plots/consumption.
    Returns:
        str: Path relatif ke gambar heatmap yang disimpan.
    """
    try:
        engine = get_db_connection()
        if not engine:
            raise Exception("Koneksi database gagal.")

        # Query untuk mengambil data yang sama seperti di PHP
        query = """
        SELECT tahun, bulan, SUM(pemakaian_kwh) as total_kwh
        FROM consumptions
        GROUP BY tahun, bulan
        ORDER BY tahun ASC, bulan ASC
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            print("Tidak ada data konsumsi yang ditemukan untuk membuat heatmap.")
            return None

        # Buat nama bulan dari angka bulan
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
            7: 'Jul', 8: 'Agu', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
        }
        df['bulan_nama'] = df['bulan'].map(month_names)

        # Pivoting data untuk format heatmap
        heatmap_data = df.pivot_table(index='tahun', columns='bulan_nama', values='total_kwh')

        # Mengurutkan kolom bulan agar sesuai urutan kalender
        all_months_ordered = [month_names[i] for i in range(1, 13)]
        heatmap_data = heatmap_data.reindex(columns=all_months_ordered)

        # Mengisi nilai NaN (bulan tanpa data) dengan 0
        heatmap_data = heatmap_data.fillna(0)

        # --- Standarisasi Kategori Warna (Tinggi, Sedang, Rendah) ---
        threshold_rendah_ke_sedang = 150000  
        threshold_sedang_ke_tinggi = 206000 

        # Menentukan batas-batas untuk normalisasi warna
        # max_kwh_data = heatmap_data.max().max()
        max_kwh_data = 400000
        
        # Jika semua nilai 0, pastikan batas atas tidak terlalu kecil
        if max_kwh_data == 0:
            bounds = [0, 1] # Hanya untuk 0, sisanya akan sama warnanya
        else:
            # Pastikan batas atas lebih besar dari nilai maksimum data
            bounds = [0, threshold_rendah_ke_sedang, threshold_sedang_ke_tinggi, max_kwh_data + 1]

        # Mendefinisikan warna secara eksplisit untuk setiap kategori
        # Urutan: Warna untuk range pertama, warna untuk range kedua, dst.
        colors = ["#2CA02C", "#FFD700", "#DC143C"] # Hijau (Rendah), Kuning (Sedang), Merah (Tinggi)
        cmap = mcolors.ListedColormap(colors)

        # Membuat objek normalisasi batas
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Buat heatmap
        plt.figure(figsize=(12, 8)) # Ukuran plot
        sns.heatmap(
            heatmap_data,
            annot=True,      # Menampilkan nilai di setiap sel
            fmt=".0f",       # Format angka tanpa desimal
            cmap=cmap,       # Menggunakan colormap kustom
            norm=norm,       # Menerapkan normalisasi BoundaryNorm di sini
            linewidths=.2,   # Garis antar sel
            linecolor='black', # Warna garis
            cbar_kws={'label': 'Total Konsumsi (kWh)'} # Label color bar
        )
        
        plt.title('Total Konsumsi Listrik per Bulan dan Tahun Seluruh Pelanggan', fontsize=16)
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Tahun', fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotasi label bulan
        plt.yticks(rotation=0) # Pastikan label tahun tidak berputar
        plt.tight_layout() # Menyesuaikan layout agar tidak terpotong

        # Simpan plot ke direktori static/plots/consumption
        plots_dir = 'static/plots/consumption'
        os.makedirs(plots_dir, exist_ok=True) # Buat direktori jika belum ada
        plot_path = os.path.join(plots_dir, output_filename)
        plt.savefig(plot_path)
        plt.close() # Penting untuk menutup plot agar tidak memakan memori

        print(f"Heatmap berhasil disimpan ke: {plot_path}")
        try:
            # This is already a valid Flask response
            return send_from_directory(plots_dir, output_filename)
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Internal Server Error: {e}"}), 500

    except Exception as e:
        print(f"Terjadi kesalahan saat membuat heatmap: {e}")
        return jsonify({"error": f"Internal Server Error: {e}"}), 500
