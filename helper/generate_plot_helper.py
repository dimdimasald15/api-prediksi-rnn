# helper/generate_plot_helper.py
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging
from utils import PLOT_FOLDER

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_plot(
    customer_id: int,
    nama: str,
    usage: List[float],
    bulan: List[int],
    tahun: List[int],
    prediksi_asli: List[float],
    jumlah_bulan: int
) -> str:
    """
    Generate prediction plot and save as image
    
    Args:
        customer_id: ID pelanggan
        nama: Nama pelanggan
        usage: Data pemakaian historis (kWh)
        bulan: Data bulan historis
        tahun: Data tahun historis
        prediksi_asli: Hasil prediksi (kWh)
        jumlah_bulan: Jumlah bulan prediksi
        
    Returns:
        str: Nama file plot yang disimpan
    """
    try:
        # Validasi input
        if len(usage) < 12 or len(bulan) < 12 or len(tahun) < 12:
            raise ValueError("Data historis harus minimal 12 bulan")
            
        if len(prediksi_asli) != jumlah_bulan:
            raise ValueError("Panjang prediksi tidak sesuai dengan jumlah bulan")

        # Buat direktori jika belum ada
        os.makedirs(PLOT_FOLDER, exist_ok=True)
        
        # Generate nama file
        plot_filename = f'prediksi_CustomerId_{customer_id}_{jumlah_bulan}_bulan.png'
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)

        # Generate labels untuk axis
        x_labels_hist = [f"{b}/{t}" for b, t in zip(bulan[-12:], tahun[-12:])]
        pred_months = []
        next_month = bulan[-1]
        next_year = tahun[-1]

        for _ in range(jumlah_bulan):
            next_month += 1
            if next_month > 12:
                next_month = 1
                next_year += 1
            pred_months.append(f"{next_month}/{next_year}")

        all_labels = x_labels_hist + pred_months

        # Buat plot
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(12), usage[-12:], label='Historis', marker='o')
        pred_line = plt.plot(
            range(12, 12 + jumlah_bulan), 
            prediksi_asli, 
            label='Prediksi', 
            marker='o', 
            linestyle='--'
        )
        pred_color = pred_line[0].get_color()
        plt.plot([11, 12], [usage[-1], prediksi_asli[0]], linestyle='--', color=pred_color)
        plt.xticks(range(12 + jumlah_bulan), all_labels, rotation=45)

        # Warna label prediksi
        ax = plt.gca()
        for i in range(jumlah_bulan):
            ax.get_xticklabels()[i + 12].set_color(pred_color)

        plt.xlabel('Bulan/Tahun')
        plt.ylabel('Pemakaian kWh')
        plt.title(f'Prediksi Pemakaian Listrik {nama.upper()}\nDalam {jumlah_bulan} Bulan Ke Depan')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Simpan plot
        fig.savefig(plot_path)
        logger.info(f"Plot berhasil disimpan: {plot_path}")
        
        return plot_filename
        
    except Exception as e:
        logger.error(f"Gagal membuat plot: {str(e)}")
        raise
    finally:
        plt.close(fig)