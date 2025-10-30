Tentu, berikut adalah penjelasan mendalam mengenai paper tersebut dalam format Markdown.

---

## Analisis Mendalam: Paper 2006.05602v1 (WS-UDA & 2ST-UDA)

[cite_start]Berdasarkan analisis mendalam saya terhadap paper `2006.05602v1` [cite: 1] dan pengalaman kita dalam proses replikasi, berikut adalah penjelasan teknis terperinci yang ditujukan untuk replikasi yang sukses.

### Ringkasan Eksekutif (Tujuan Paper)

[cite_start]Paper ini bertujuan untuk memecahkan masalah **Multi-Source Unsupervised Domain Adaptation (MS-UDA)**[cite: 10].

* **Skenario:** Kita memiliki beberapa domain "sumber" (misal: *review* buku, DVD, elektronik) di mana kita memiliki data latih **berlabel** (positif/negatif). [cite_start]Kita juga memiliki satu domain "target" (misal: *review* dapur) di mana kita hanya punya data **tanpa label**[cite: 10, 149].
* **Tujuan:** Membuat *classifier* sentimen yang memiliki akurasi tinggi di domain target, meskipun tidak pernah dilatih dengan label dari domain tersebut.
* **Solusi:** Paper ini mengusulkan dua *framework*:
    1.  **WS-UDA (Weighting Scheme based UDA):** Framework utama yang melatih model "Shared-Private" secara adversarial. [cite_start]Kuncinya adalah menggunakan **Discriminator (D)** untuk (a) memisahkan fitur [cite: 16] [cite_start]dan (b) menghitung "bobot kepercayaan" ($w$) untuk setiap domain sumber saat memprediksi data target[cite: 13, 15].
    2.  [cite_start]**2ST-UDA (Two-Stage Training based UDA):** Framework kedua yang mengambil *pseudo-label* (hasil tebakan) dari WS-UDA, lalu menggunakannya untuk melatih *extractor* khusus untuk domain target ($E_t$)[cite: 14].

---

### Arsitektur Model Inti (Gambar 1)

[cite_start]Untuk mereplikasi **WS-UDA**, kita harus membangun 4 komponen utama yang saling terkait [cite: 141-145]. [cite_start]Untuk dataset Amazon (input 5000d BoW), semua komponen ini adalah model MLP sederhana[cite: 278].



1.  **$E_s$ (Shared Extractor):**
    * **Apa:** Satu *feature extractor* (MLP) yang digunakan oleh *semua* domain (sumber + target).
    * [cite_start]**Tujuan:** Menangkap fitur yang **domain-invariant** (fitur umum yang artinya sama di semua domain, misal: "jelek", "luar biasa")[cite: 141].
    * **Output:** Vektor fitur bersama $z_s$.

2.  **$\{E_{p_j}\}_{j=1}^{K}$ (Private Extractors):**
    * **Apa:** $K$ *feature extractor* (MLP) yang **terpisah**. [cite_start]Ada satu extractor privat untuk *setiap* domain sumber ($j$)[cite: 142]. (Misal: $E_{p\_buku}$, $E_{p\_dvd}$, $E_{p\_elektronik}$).
    * [cite_start]**Tujuan:** Menangkap fitur yang **domain-specific** (fitur unik yang maknanya berbeda antar domain, misal: "cepat" positif untuk elektronik, tapi negatif untuk baterai)[cite: 142, 143].
    * **Output:** Vektor fitur privat $z_{p_j}$.

3.  **$C$ (Sentiment Classifier):**
    * **Apa:** Satu *classifier* (MLP).
    * [cite_start]**Tujuan:** Memprediksi sentimen (positif/negatif)[cite: 144].
    * **Input:** Gabungan dari fitur bersama dan fitur privat: $\text{concat}(z_s, z_{p_j})$.
    * [cite_start]**Output:** Logit sentimen $\hat{c}_j$ (prediksi sentimen dari "sudut pandang" domain $j$)[cite: 144].

4.  **$D$ (Domain Discriminator):**
    * **Apa:** Satu *classifier* domain (MLP).
    * **Tujuan (Ganda):**
        1.  [cite_start]**Saat Training:** Dilatih untuk menebak domain asal data (misal: "ini buku", "ini dvd", "ini elektronik", atau "ini dapur") berdasarkan fitur $z_s$ atau $z_p$[cite: 143].
        2.  [cite_start]**Saat Evaluasi:** Digunakan sebagai **penghitung bobot ($w$)**[cite: 15, 209]. [cite_start]Saat diberi $z_s$ dari data target, $D$ akan memberi tahu "seberapa mirip" fitur ini dengan setiap domain sumber[cite: 15, 145].

---

### Metodologi 1: WS-UDA (Algoritma 1)

Ini adalah inti dari replikasi. [cite_start]*Training* ini bersifat *adversarial* (saling melawan) dan dibagi menjadi dua tahap yang diulang-ulang di setiap *batch* [cite: 204-207].

#### Tahap 1: Latih Discriminator (D)

* [cite_start]**Tujuan:** Membuat `D` menjadi "kritikus" yang pintar dalam menebak domain[cite: 205].
* **Langkah:**
    1.  **Bekukan (freeze)** $E_s$, $\{E_p\}$, dan $C$.
    2.  [cite_start]Ambil data dari **semua domain** (K sumber + 1 target, disebut $\mathcal{U}$)[cite: 89, 111].
    3.  [cite_start]Hitung **Loss $\mathcal{L}_D$** (berdasarkan Persamaan 2 dan Algoritma 1 [cite: 109-126]):
        * `loss_D_s`: $D$ harus menebak domain dengan benar dari $z_s = E_s(x)$. [cite_start]Ini berlaku untuk data *source* dan *target*[cite: 116].
        * `loss_D_p`: $D$ harus menebak domain dengan benar dari $z_p = E_{p_j}(x)$. [cite_start]Ini hanya berlaku untuk data *source* (karena target tidak punya $E_p$)[cite: 120, 122].
    4.  [cite_start]`loss_D = loss_D_s + loss_D_p`[cite: 125].
    5.  [cite_start]Lakukan *backpropagation* hanya pada `D`[cite: 126].
    6.  [cite_start]Ulangi langkah ini $n_{critic}$ kali (misal, 5 kali)[cite: 90, 106].

#### Tahap 2: Latih Model Utama (E_s, E_p, C)

* [cite_start]**Tujuan:** (1) Jago menebak sentimen [cite: 158][cite_start], (2) $E_s$ "menipu" $D$ [cite: 161, 202][cite_start], dan (3) $E_p$ "membantu" $D$[cite: 179].
* **Langkah:**
    1.  **Bekukan (freeze)** `D`.
    2.  [cite_start]Ambil data *source* ($\mathcal{S}$) [cite: 132-135] [cite_start]dan data *target* ($\mathcal{T}$) [cite: 159-161].
    3.  Hitung 3 *loss* terpisah:
        * **Loss $\mathcal{L}_C$ (Sentiment):**
            * [cite_start]Gunakan data *source* ($x_s, y_s$)[cite: 135].
            * $z_s = E_s(x_s)$, $z_p = E_{p_j}(x_s)$ (untuk domain $j$).
            * $\hat{y} = C(\text{concat}(z_s, z_p))$.
            * `loss_C` adalah *Cross-Entropy* antara $\hat{y}$ dan label asli $y_s$. [cite_start](Ini adalah $\mathcal{L}_C$ dari Persamaan 3 [cite: 187] [cite_start]dan Algoritma 1 [cite: 158]).
        * **Loss $\mathcal{L}_{S\_adv}$ (Adversarial Shared):**
            * [cite_start]Gunakan data *source* ($x_s$) dan *target* ($x_t$) (semua domain $\mathcal{U}$) [cite: 159-161].
            * $z_s = E_s(x_s, x_t)$.
            * **Terapkan GRL:** $z_{s\_adv} = \text{GRL}(z_s)$.
            * $\hat{d} = D(z_{s\_adv})$.
            * `loss_Es_adv` adalah *Cross-Entropy* antara $\hat{d}$ dan label domain *asli*. [cite_start]Karena GRL membalik gradien, meminimalkan *loss* ini sebenarnya melatih $E_s$ untuk *memaksimalkan* kesalahan $D$[cite: 161, 202]. $E_s$ dilatih untuk "menipu" $D$.
        * **Loss $\mathcal{L}_{P\_priv}$ (Private Domain):**
            * Gunakan data *source* ($x_s$).
            * $z_p = E_{p_j}(x_s)$ (TANPA GRL).
            * $\hat{d} = D(z_p)$.
            * `loss_domain_private` adalah *Cross-Entropy* antara $\hat{d}$ dan label domain asli $j$.
            * [cite_start]Ini melatih $E_p$ untuk "membantu" $D$ menebak domain dengan benar, memaksanya menyerap fitur *domain-specific* (Sesuai Persamaan 4 [cite: 176, 179]).
    4.  Gabungkan *loss* (dengan bobot $\alpha, \beta, \gamma$):
        [cite_start]`loss_main = (alpha * loss_C) + (beta * loss_Es_adv) + (gamma * loss_domain_private)`[cite: 162].
    5.  [cite_start]Lakukan *backpropagation* pada $E_s, \{E_p\}, C$[cite: 162].
    6.  Terapkan *gradient clipping* untuk stabilitas.

#### Proses Evaluasi (Inference) WS-UDA

[cite_start]Ini adalah implementasi dari "Weighting Scheme" [cite: 209-211].

1.  Ambil data target $x_t$.
2.  Dapatkan fitur bersama: $z_s = E_s(x_t)$.
3.  [cite_start]**Hitung Bobot ($w$):** Umpankan $z_s$ ke `D`[cite: 209]. `D` akan menghasilkan $K+1$ logits (misal, 4: buku, dvd, elektro, dapur).
4.  Ambil **hanya $K$ logits pertama** (logits domain *sumber*).
5.  [cite_start]Terapkan **Softmax** (disebut "Normalize" di paper [cite: 145, 211]) pada $K$ logits tersebut untuk mendapatkan bobot $w$. $w$ adalah vektor dengan $K$ angka (misal, `[0.7, 0.1, 0.2]`) yang menunjukkan seberapa "mirip" $x_t$ dengan setiap domain sumber.
6.  **Hitung Prediksi Tertimbang:**
    * Buat $K$ prediksi sentimen, satu untuk setiap "sudut pandang" domain sumber:
        * $\hat{c}_1 = C(\text{concat}(z_s, E_{p\_buku}(x_t)))$
        * $\hat{c}_2 = C(\text{concat}(z_s, E_{p\_dvd}(x_t)))$
        * $\hat{c}_3 = C(\text{concat}(z_s, E_{p\_elektronik}(x_t)))$
    * [cite_start]Kalikan setiap prediksi dengan bobotnya: $\hat{y}_{final} = (w_1 \cdot \hat{c}_1) + (w_2 \cdot \hat{c}_2) + (w_3 \cdot \hat{c}_3)$ (Sesuai Persamaan 6 [cite: 211]).
7.  Label akhir adalah $\text{argmax}(\hat{y}_{final})$.

---

### Metodologi 2: 2ST-UDA (Algoritma 2)

[cite_start]Framework ini adalah langkah *tambahan* setelah WS-UDA selesai[cite: 214]. [cite_start]Tujuannya adalah menggunakan *pseudo-label* untuk melatih *extractor* khusus target[cite: 257].

1.  [cite_start]**Inisialisasi:** Muat model $E_s, \{E_p\}, C, D$ yang sudah dilatih dari WS-UDA[cite: 216]. [cite_start]Buat satu *extractor* baru: $E_t$ (Target Private Extractor)[cite: 217].
2.  **Labeling Berbasis Kepercayaan:**
    * [cite_start]Tetapkan ambang batas kepercayaan $\Delta$ (misal, 0.98)[cite: 217, 260].
    * Untuk setiap data target $x_t$:
        * [cite_start]Dapatkan pseudo-label dari "View 1" (Source): $\hat{y}_S = \text{WS-UDA}(x_t)$ (disebut `labeling(...)` di algoritma [cite: 220]).
        * [cite_start]Dapatkan pseudo-label dari "View 2" (Target): $\hat{y}_T = C(\text{concat}(E_s(x_t), E_t(x_t)))$[cite: 221].
    * [cite_start]Jika $\hat{y}_S == \hat{y}_T$ DAN *confidence* (probabilitas softmax) dari $\hat{y}_S$ > $\Delta$ [cite: 222, 269][cite_start], maka tambahkan $(x_t, \hat{y}_S)$ ke set data latih baru $\mathcal{L}$[cite: 223].
3.  [cite_start]**Training $E_t$:** Latih $E_t$ menggunakan data $\mathcal{L}$ yang baru dibuat [cite: 224-226].
4.  [cite_start]**Iterasi:** Ulangi langkah 2-3, sambil menurunkan $\Delta$ secara perlahan (misal, $\Delta = \Delta - 0.02$)[cite: 241, 261]. Ini secara bertahap menambahkan lebih banyak data (yang kurang diyakini) ke dalam set $\mathcal{L}$.
5.  [cite_start]**Finetune:** Setelah iterasi selesai (misal, $\Delta \le 0.5$ [cite: 247][cite_start]), latih $E_t$ sekali lagi menggunakan semua data $\mathcal{L}$ yang terkumpul [cite: 248-253].

---

### Data dan Poin Kunci Replikasi

Ini adalah bagian paling penting yang kita temukan bersama.

* **Dataset 1: Amazon Review**
    * [cite_start]**Data:** Inputnya BUKAN teks mentah[cite: 277]. [cite_start]Ini adalah file `.review` yang berisi vektor **Bag-of-Words (BoW) 5000-dimensi**[cite: 278].
    * **Bug Indeks 1-Based:** File-file ini menggunakan *indexing* berbasis 1 (fitur 1 sampai 5000). Saat mem-parsing, setiap indeks harus **dikurangi 1** untuk menjadi 0-4999 agar pas dengan *tensor* PyTorch.
    * [cite_start]**Model:** Karena inputnya adalah vektor BoW, semua *extractor* ($E_s, E_p, E_t$) dan *classifier* ($C, D$) **harus berupa MLP**[cite: 278].
    * **Normalisasi Input:** Sangat disarankan untuk menerapkan L2-Normalize pada vektor 5000d di awal `forward` pass $E_s$ dan $E_p$ untuk menstabilkan *training*.
    * [cite_start]**Target (Kit): 87.66%** [cite: 307]

* **Dataset 2: FDU-MTL**
    * [cite_start]**Data:** Ini adalah **teks mentah** (*raw review texts*)[cite: 281].
    * [cite_start]**Model:** Di sini, *extractor* ($E_s, E_p, E_t$) harus berupa model sekuensial seperti **LSTM**[cite: 282]. $C$ dan $D$ tetap MLP yang menerima output dari LSTM.
    * [cite_start]**Target (Avg): 87.1%** (WS-UDA) [cite: 325]

* **Hyperparameter Kunci:**
    * [cite_start]**Batch Size:** Paper menggunakan **8**[cite: 292].
    * [cite_start]**Learning Rate:** Paper menggunakan **0.0001** (1e-4)[cite: 292]. Berdasarkan pengalaman kita, menggunakan LR yang sedikit lebih rendah untuk `D` (misal, 5e-5) dan *weight decay* dapat membantu stabilitas.
    * **Gradient Clipping:** Wajib digunakan di *main step* (Tahap 2) untuk mencegah "ledakan" gradien.
    * [cite_start]**Training:** Paper menggunakan *early stopping* pada *validation set*[cite: 293]. Pendekatan kita untuk menyimpan model terbaik berdasarkan akurasi validasi per epoch adalah implementasi yang valid dari ini.

