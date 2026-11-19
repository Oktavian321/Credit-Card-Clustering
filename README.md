# Credit Card Clustering

Proyek *unsupervised learning* untuk mengelompokkan nasabah kartu kredit berdasarkan perilaku penggunaan kartu, menggunakan **KMeans**. Jumlah cluster ditentukan dengan *elbow method*, lalu hasilnya dievaluasi dan diprofilkan menjadi segmen nasabah yang bisa diinterpretasikan.

## Ringkasan

- **Tujuan:** segmentasi nasabah kartu kredit berdasarkan perilaku transaksi
- **Data:** 8.950 nasabah, 18 kolom, tanpa duplikat. Missing value ada pada `MINIMUM_PAYMENTS` (313) dan `CREDIT_LIMIT` (1)
- **Tantangan utama:** distribusi fitur miring dan banyak outlier (skewness `MINIMUM_PAYMENTS` 13,6; `ONEOFF_PURCHASES` 10,0; `PURCHASES` 8,1)
- **Metrik utama:** Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index
- **Hasil:** 5 cluster (k = 5) dengan Silhouette 0,226; Davies-Bouldin 1,567; Calinski-Harabasz 2188,68

## Dataset

[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) dari Kaggle (`CC GENERAL.csv`), berisi ringkasan penggunaan kartu kredit selama 6 bulan terakhir, seperti saldo, total belanja, penarikan tunai, limit kredit, pembayaran, dan persentase pelunasan penuh.

## Alur Kerja

1. **Data understanding:** `head`, `info`, `describe`, cek missing value dan duplikat.
2. **EDA:** histogram semua fitur numerik beserta skewness, boxplot untuk melihat outlier, dan jumlah nilai unik per kolom.
3. **Data cleaning:**
   - `MINIMUM_PAYMENTS` yang kosong diisi 0, setelah dicek bahwa pada baris tersebut `PAYMENTS` umumnya bernilai 0.
   - `CREDIT_LIMIT` yang kosong (1 baris) diisi median.
   - Kolom `CUST_ID` dihapus karena tidak informatif.
   - `CASH_ADVANCE_FREQUENCY` dibatasi maksimum 1 (`clip`) karena nilainya berupa frekuensi.
4. **Transformasi log:** `np.log1p` pada 11 fitur yang skewness-nya berat. Contoh: `MINIMUM_PAYMENTS` 13,62 menjadi -1,29, `ONEOFF_PURCHASES` 10,05 menjadi 0,19, dan `PURCHASES` 8,14 menjadi -0,76.
5. **Scaling:** `StandardScaler` pada seluruh fitur.
6. **Penentuan jumlah cluster:** *elbow method* (`KElbowVisualizer`, k = 2 sampai 11, metrik distortion) menghasilkan **k = 5**.
7. **Modeling:** KMeans (`n_clusters=5`, `n_init=10`, `random_state=42`).
8. **Evaluasi:** Silhouette, Davies-Bouldin, dan Calinski-Harabasz.
9. **Profiling cluster:** rata-rata fitur per cluster dihitung pada data asli agar nilainya mudah dibaca, lalu setiap cluster diberi nama segmen.

## Hasil

### Evaluasi Cluster

| Metrik | Nilai | Arah terbaik |
|---|---|---|
| Silhouette Score | 0.226 | Makin tinggi makin baik |
| Davies-Bouldin Index | 1.567 | Makin kecil makin baik |
| Calinski-Harabasz Index | 2188.68 | Makin besar makin baik |

### Profil Segmen

| Cluster | Segmen | Jumlah | Persen | Karakteristik |
|---|---|---|---|---|
| 0 | Penarik Tunai | 2.444 | 27,3% | Hampir tidak berbelanja, kartu dipakai untuk tarik tunai bernilai besar. Saldo tinggi dan jarang melunasi tagihan penuh (`PRC_FULL_PAYMENT` 0,03). |
| 2 | Big Spender | 1.965 | 22,0% | Belanja dan jumlah transaksi tertinggi, dominan pembelian sekali bayar, limit tertinggi, hampir tidak pernah tarik tunai. |
| 1 | Pembeli Cicilan | 1.883 | 21,0% | Rutin berbelanja dan dominan memakai cicilan, jarang tarik tunai, limit terendah, paling sering melunasi tagihan penuh (0,31). |
| 4 | Heavy User | 1.388 | 15,5% | Belanja sekaligus tarik tunai dalam skala besar, saldo dan pembayaran tertinggi, tetapi jarang melunasi tagihan penuh (0,04). |
| 3 | Jarang Digunakan | 1.270 | 14,2% | Hampir tidak aktif di semua jenis transaksi, saldo, pembayaran, dan minimum payment terendah. |

## Keterbatasan

- Hanya memakai KMeans, dan jumlah cluster ditentukan lewat elbow method saja.
- `MINIMUM_PAYMENTS` yang kosong diisi 0 berdasarkan asumsi bahwa nasabah tersebut tidak melakukan pembayaran.
- Beberapa fitur miring setelah transformasi log (`PRC_FULL_PAYMENT` 1,94; `BALANCE_FREQUENCY` -2,02; `TENURE` -2,94).
- Nama segmen bersifat interpretatif karena tidak ada label asli sebagai pembanding.

## Rencana Pengembangan

- Menentukan jumlah cluster dengan metrik tambahan (misalnya Silhouette per nilai k).
- Membandingkan algoritma lain (Gaussian Mixture, Hierarchical Clustering, DBSCAN).
- Visualisasi cluster dengan PCA atau t-SNE.
- Menangani fitur yang masih miring dan mencoba seleksi fitur.
- Menyusun rekomendasi strategi untuk setiap segmen.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Yellowbrick
