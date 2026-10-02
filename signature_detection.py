import cv2
import os
import numpy as np


# ============================================================
# 1. PENGATURAN FOLDER
# ============================================================

IMAGE_FOLDER = "gambar"

PRESENT_FOLDER = os.path.join(
    IMAGE_FOLDER,
    "present"
)

ABSENT_FOLDER = os.path.join(
    IMAGE_FOLDER,
    "absent"
)

OUTPUT_FOLDER = "hasil"

GRAY_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "grayscale"
)

GLOBAL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "threshold_global"
)

ADAPTIVE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "threshold_adaptive"
)

OPENING_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "opening"
)

CLOSING_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "closing"
)

COMPARISON_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "comparison"
)


# ============================================================
# 2. MEMBUAT FOLDER
# ============================================================

folders = [
    PRESENT_FOLDER,
    ABSENT_FOLDER,
    GRAY_FOLDER,
    GLOBAL_FOLDER,
    ADAPTIVE_FOLDER,
    OPENING_FOLDER,
    CLOSING_FOLDER,
    COMPARISON_FOLDER
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# 3. PARAMETER
# ============================================================

# Global Threshold
GLOBAL_THRESHOLD = 127

# Adaptive Threshold
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Kernel Morphology
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 3)
)

# Batas foreground
FOREGROUND_LIMIT = 2.0


# ============================================================
# 4. UKURAN COMPARISON
# ============================================================

LEBAR = 500
TINGGI = 300

# Jarak vertikal antar baris
JARAK_VERTIKAL = 30

# Jarak horizontal antar gambar
JARAK_HORIZONTAL = 40

# Lebar keseluruhan canvas
LEBAR_CANVAS = (
    LEBAR * 2 +
    JARAK_HORIZONTAL
)


# ============================================================
# 5. FUNGSI MEMBUAT GAMBAR ABSENT
# ============================================================

def buat_gambar_absent(image):

    """
    Membuat gambar tanpa tanda tangan
    dari gambar yang berada di folder PRESENT.
    """

    # --------------------------------------------------------
    # Grayscale sementara
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Adaptive Threshold untuk mencari area tanda tangan
    # --------------------------------------------------------

    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    # --------------------------------------------------------
    # Membersihkan mask
    # --------------------------------------------------------

    kernel_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_mask
    )

    mask = cv2.dilate(
        mask,
        kernel_mask,
        iterations=1
    )

    # --------------------------------------------------------
    # Menghapus area tanda tangan
    # --------------------------------------------------------

    absent_image = cv2.inpaint(
        image,
        mask,
        3,
        cv2.INPAINT_TELEA
    )

    return absent_image


# ============================================================
# 6. MENCARI FILE PRESENT
# ============================================================

present_files = []

for nama_file in os.listdir(PRESENT_FOLDER):

    if nama_file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        present_files.append(
            nama_file
        )

present_files.sort()


# ============================================================
# 7. MEMBUAT GAMBAR ABSENT
# ============================================================

print()
print("=" * 70)
print("TAHAP 1 - MEMBUAT GAMBAR ABSENT")
print("=" * 70)

if len(present_files) == 0:

    print()
    print("Tidak ada gambar di folder:")
    print(os.path.abspath(PRESENT_FOLDER))

    input("\nTekan ENTER untuk keluar...")
    exit()


for nama_file in present_files:

    print()
    print("Memproses:")
    print(nama_file)

    input_path = os.path.join(
        PRESENT_FOLDER,
        nama_file
    )

    image = cv2.imread(
        input_path
    )

    if image is None:

        print("Gagal membaca gambar.")
        continue

    # Membuat gambar tanpa tanda tangan
    absent_image = buat_gambar_absent(
        image
    )

    # Menyimpan gambar absent
    output_path = os.path.join(
        ABSENT_FOLDER,
        nama_file
    )

    cv2.imwrite(
        output_path,
        absent_image
    )

    print("Disimpan ke:")
    print(output_path)


print()
print("Pembuatan gambar ABSENT selesai.")


# ============================================================
# 8. FUNGSI MEMBERI JUDUL PADA GAMBAR
# ============================================================

def beri_keterangan(image, judul):

    # Jika gambar grayscale
    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    tinggi, lebar = image.shape[:2]

    # Tinggi area judul
    tinggi_judul = 45

    # Membuat area putih
    judul_area = np.ones(
        (
            tinggi_judul,
            lebar,
            3
        ),
        dtype=np.uint8
    ) * 255

    # Menulis judul
    cv2.putText(
        judul_area,
        judul,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Gabungkan judul dengan gambar
    hasil = np.vstack([
        judul_area,
        image
    ])

    return hasil


# ============================================================
# 9. FUNGSI RESIZE
# ============================================================

def resize_image(image):

    return cv2.resize(
        image,
        (LEBAR, TINGGI)
    )


# ============================================================
# 10. FUNGSI SPACER VERTIKAL
# ============================================================

def buat_spacer_vertikal():

    return np.ones(
        (
            JARAK_VERTIKAL,
            LEBAR_CANVAS,
            3
        ),
        dtype=np.uint8
    ) * 255


# ============================================================
# 11. FUNGSI MEMBUAT AREA NAMA FILE
# ============================================================

def buat_nama_area(nama_file):

    tinggi_nama = 65

    nama_area = np.ones(
        (
            tinggi_nama,
            LEBAR_CANVAS,
            3
        ),
        dtype=np.uint8
    ) * 255

    cv2.putText(
        nama_area,
        "NAMA FILE:",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        nama_area,
        nama_file,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return nama_area


# ============================================================
# 12. FUNGSI MEMBUAT BARIS DUA GAMBAR
# ============================================================

def buat_baris_dua_gambar(
    gambar_kiri,
    gambar_kanan
):

    # Spacer horizontal
    spacer_horizontal = np.ones(
        (
            TINGGI,
            JARAK_HORIZONTAL,
            3
        ),
        dtype=np.uint8
    ) * 255

    # Gabungkan:
    # gambar kiri + jarak + gambar kanan
    baris = np.hstack([
        gambar_kiri,
        spacer_horizontal,
        gambar_kanan
    ])

    return baris


# ============================================================
# 13. FUNGSI MEMBUAT GRAYSCALE DI TENGAH
# ============================================================

def buat_baris_tengah(gambar):

    canvas = np.ones(
        (
            TINGGI,
            LEBAR_CANVAS,
            3
        ),
        dtype=np.uint8
    ) * 255

    posisi_x = (
        (LEBAR_CANVAS - LEBAR) // 2
    )

    canvas[
        :,
        posisi_x:
        posisi_x + LEBAR
    ] = gambar

    return canvas


# ============================================================
# 14. FUNGSI INFORMASI FOREGROUND
# ============================================================

def buat_info_foreground(
    foreground_pixels,
    total_pixels,
    foreground_percentage
):

    tinggi_info = 125

    info_area = np.ones(
        (
            tinggi_info,
            LEBAR_CANVAS,
            3
        ),
        dtype=np.uint8
    ) * 255

    # Judul
    cv2.putText(
        info_area,
        "HASIL PERHITUNGAN FOREGROUND",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Foreground pixel
    cv2.putText(
        info_area,
        f"Foreground Pixel : {foreground_pixels}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    # Total pixel
    cv2.putText(
        info_area,
        f"Total Pixel      : {total_pixels}",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    # Persentase
    cv2.putText(
        info_area,
        f"Foreground (%)   : {foreground_percentage:.2f}%",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return info_area


# ============================================================
# 15. DATA HASIL
# ============================================================

hasil_data = []


# ============================================================
# 16. FUNGSI PROSES SATU GAMBAR
# ============================================================

def proses_gambar(
    nama_file,
    label,
    folder_input
):

    print()
    print("=" * 70)
    print("MEMPROSES GAMBAR")
    print("=" * 70)

    print("Nama File :", nama_file)
    print("Label     :", label)

    # ========================================================
    # A. MEMBACA GAMBAR
    # ========================================================

    image_path = os.path.join(
        folder_input,
        nama_file
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print("Gagal membaca gambar.")
        return

    # ========================================================
    # B. GRAYSCALE
    # ========================================================

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_path = os.path.join(
        GRAY_FOLDER,
        label.lower() + "_" + nama_file
    )

    cv2.imwrite(
        gray_path,
        grayscale
    )

    # ========================================================
    # C. GLOBAL THRESHOLD
    # ========================================================

    _, threshold_global = cv2.threshold(
        grayscale,
        GLOBAL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    global_path = os.path.join(
        GLOBAL_FOLDER,
        label.lower() + "_" + nama_file
    )

    cv2.imwrite(
        global_path,
        threshold_global
    )

    # ========================================================
    # D. ADAPTIVE THRESHOLD
    # ========================================================

    threshold_adaptive = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    adaptive_path = os.path.join(
        ADAPTIVE_FOLDER,
        label.lower() + "_" + nama_file
    )

    cv2.imwrite(
        adaptive_path,
        threshold_adaptive
    )

    # ========================================================
    # E. MORPHOLOGICAL OPENING
    # ========================================================

    opening = cv2.morphologyEx(
        threshold_adaptive,
        cv2.MORPH_OPEN,
        kernel
    )

    opening_path = os.path.join(
        OPENING_FOLDER,
        label.lower() + "_" + nama_file
    )

    cv2.imwrite(
        opening_path,
        opening
    )

    # ========================================================
    # F. MORPHOLOGICAL CLOSING
    # ========================================================

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    closing_path = os.path.join(
        CLOSING_FOLDER,
        label.lower() + "_" + nama_file
    )

    cv2.imwrite(
        closing_path,
        closing
    )

    # ========================================================
    # G. HITUNG FOREGROUND PIXEL
    # ========================================================

    foreground_pixels = cv2.countNonZero(
        closing
    )

    total_pixels = (
        closing.shape[0] *
        closing.shape[1]
    )

    foreground_percentage = (
        foreground_pixels /
        total_pixels
    ) * 100

    # ========================================================
    # H. ATURAN DETEKSI
    # ========================================================

    if foreground_percentage >= FOREGROUND_LIMIT:

        hasil_deteksi = "SIGNATURE PRESENT"

    else:

        hasil_deteksi = "SIGNATURE ABSENT"

    # ========================================================
    # I. EVALUASI
    # ========================================================

    if label == "PRESENT":

        if hasil_deteksi == "SIGNATURE PRESENT":

            status = "BENAR"

        else:

            status = "SALAH"

    else:

        if hasil_deteksi == "SIGNATURE ABSENT":

            status = "BENAR"

        else:

            status = "SALAH"

    # ========================================================
    # J. SIMPAN DATA
    # ========================================================

    hasil_data.append({

        "nama_file": nama_file,

        "label": label,

        "foreground": foreground_pixels,

        "total": total_pixels,

        "persentase": foreground_percentage,

        "deteksi": hasil_deteksi,

        "status": status

    })

    # ========================================================
    # K. MEMBUAT GAMBAR COMPARISON
    #
    # URUTAN:
    #
    #                GRAYSCALE
    #
    # GLOBAL THRESHOLD | ADAPTIVE THRESHOLD
    #
    # OPENING          | CLOSING
    #
    # HASIL FOREGROUND
    # ========================================================

    # --------------------------------------------------------
    # Nama file
    # --------------------------------------------------------

    nama_area = buat_nama_area(
        nama_file
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gambar_grayscale = beri_keterangan(
        grayscale,
        "GRAYSCALE"
    )

    gambar_grayscale = resize_image(
        gambar_grayscale
    )

    # --------------------------------------------------------
    # GLOBAL THRESHOLD
    # --------------------------------------------------------

    gambar_global = beri_keterangan(
        threshold_global,
        "GLOBAL THRESHOLD"
    )

    gambar_global = resize_image(
        gambar_global
    )

    # --------------------------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    gambar_adaptive = beri_keterangan(
        threshold_adaptive,
        "ADAPTIVE THRESHOLD"
    )

    gambar_adaptive = resize_image(
        gambar_adaptive
    )

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------

    gambar_opening = beri_keterangan(
        opening,
        "OPENING"
    )

    gambar_opening = resize_image(
        gambar_opening
    )

    # --------------------------------------------------------
    # CLOSING
    # --------------------------------------------------------

    gambar_closing = beri_keterangan(
        closing,
        "CLOSING"
    )

    gambar_closing = resize_image(
        gambar_closing
    )

    # ========================================================
    # L. GRAYSCALE DI ATAS DAN DI TENGAH
    # ========================================================

    baris_grayscale = buat_baris_tengah(
        gambar_grayscale
    )

    # ========================================================
    # M. GLOBAL + ADAPTIVE
    # ========================================================

    baris_global_adaptive = buat_baris_dua_gambar(
        gambar_global,
        gambar_adaptive
    )

    # ========================================================
    # N. OPENING + CLOSING
    # ========================================================

    baris_opening_closing = buat_baris_dua_gambar(
        gambar_opening,
        gambar_closing
    )

    # ========================================================
    # O. INFORMASI FOREGROUND
    # ========================================================

    info_area = buat_info_foreground(
        foreground_pixels,
        total_pixels,
        foreground_percentage
    )

    # ========================================================
    # P. SPACER
    # ========================================================

    spacer = buat_spacer_vertikal()

    # ========================================================
    # Q. GABUNGKAN COMPARISON
    # ========================================================

    comparison = np.vstack([

        # Nama file
        nama_area,

        spacer,

        # Grayscale
        baris_grayscale,

        spacer,

        # Global + Adaptive
        baris_global_adaptive,

        spacer,

        # Opening + Closing
        baris_opening_closing,

        spacer,

        # Foreground
        info_area

    ])

    # ========================================================
    # R. SIMPAN COMPARISON
    # ========================================================

    comparison_name = (
        label.lower()
        + "_"
        + nama_file
    )

    comparison_path = os.path.join(
        COMPARISON_FOLDER,
        comparison_name
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    # ========================================================
    # S. TAMPILKAN HASIL
    # ========================================================

    print()
    print("Foreground Pixel   :", foreground_pixels)

    print(
        "Total Pixel        :",
        total_pixels
    )

    print(
        "Foreground (%)     :",
        f"{foreground_percentage:.2f}%"
    )

    print(
        "Hasil Deteksi      :",
        hasil_deteksi
    )

    print(
        "Status             :",
        status
    )

    print(
        "Comparison         :",
        comparison_path
    )


# ============================================================
# 17. PROSES GAMBAR PRESENT
# ============================================================

print()
print("=" * 70)
print("TAHAP 2 - PROSES GAMBAR PRESENT")
print("=" * 70)

for nama_file in present_files:

    proses_gambar(
        nama_file,
        "PRESENT",
        PRESENT_FOLDER
    )


# ============================================================
# 18. MENCARI GAMBAR ABSENT
# ============================================================

absent_files = []

for nama_file in os.listdir(
    ABSENT_FOLDER
):

    if nama_file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        absent_files.append(
            nama_file
        )

absent_files.sort()


# ============================================================
# 19. PROSES GAMBAR ABSENT
# ============================================================

print()
print("=" * 70)
print("TAHAP 3 - PROSES GAMBAR ABSENT")
print("=" * 70)

for nama_file in absent_files:

    proses_gambar(
        nama_file,
        "ABSENT",
        ABSENT_FOLDER
    )


# ============================================================
# 20. MEMBUAT FILE TXT
# ============================================================

txt_path = os.path.join(
    OUTPUT_FOLDER,
    "hasil_deteksi_signature.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 150 + "\n"
    )

    file.write(
        "HASIL DETEKSI KEBERADAAN TANDA TANGAN KEPALA SEKOLAH\n"
    )

    file.write(
        "=" * 150 + "\n\n"
    )

    # Header tabel
    file.write(
        f"{'No':<4}"
        f"{'Nama File':<45}"
        f"{'Label':<10}"
        f"{'Foreground':<15}"
        f"{'Total Pixel':<15}"
        f"{'FG (%)':<10}"
        f"{'Hasil Deteksi':<22}"
        f"{'Status':<10}\n"
    )

    file.write(
        "-" * 150 + "\n"
    )

    # Isi tabel
    for nomor, data in enumerate(
        hasil_data,
        start=1
    ):

        file.write(
            f"{nomor:<4}"
            f"{data['nama_file']:<45}"
            f"{data['label']:<10}"
            f"{data['foreground']:<15}"
            f"{data['total']:<15}"
            f"{data['persentase']:<10.2f}"
            f"{data['deteksi']:<22}"
            f"{data['status']:<10}\n"
        )

    # Keterangan
    file.write("\n")

    file.write(
        "=" * 150 + "\n"
    )

    file.write(
        "KETERANGAN\n"
    )

    file.write(
        "=" * 150 + "\n"
    )

    file.write(
        "PRESENT = gambar yang memiliki tanda tangan.\n"
    )

    file.write(
        "ABSENT = gambar yang tidak memiliki tanda tangan.\n"
    )

    file.write(
        "Foreground Pixel = jumlah piksel foreground setelah segmentasi.\n"
    )

    file.write(
        "Foreground (%) = persentase foreground terhadap seluruh piksel.\n"
    )

    file.write(
        f"Batas deteksi = {FOREGROUND_LIMIT:.2f}%.\n"
    )

    file.write(
        "Jika Foreground (%) >= batas, hasil = SIGNATURE PRESENT.\n"
    )

    file.write(
        "Jika Foreground (%) < batas, hasil = SIGNATURE ABSENT.\n"
    )


# ============================================================
# 21. MENGHITUNG AKURASI
# ============================================================

jumlah_benar = 0
jumlah_salah = 0

for data in hasil_data:

    if data["status"] == "BENAR":

        jumlah_benar += 1

    else:

        jumlah_salah += 1


total_data = len(
    hasil_data
)


if total_data > 0:

    akurasi = (
        jumlah_benar /
        total_data
    ) * 100

else:

    akurasi = 0


# ============================================================
# 22. TAMBAHKAN AKURASI KE TXT
# ============================================================

with open(
    txt_path,
    "a",
    encoding="utf-8"
) as file:

    file.write("\n")

    file.write(
        "=" * 150 + "\n"
    )

    file.write(
        "EVALUASI SISTEM\n"
    )

    file.write(
        "=" * 150 + "\n"
    )

    file.write(
        f"Total data       : {total_data}\n"
    )

    file.write(
        f"Prediksi benar   : {jumlah_benar}\n"
    )

    file.write(
        f"Prediksi salah   : {jumlah_salah}\n"
    )

    file.write(
        f"Akurasi          : {akurasi:.2f}%\n"
    )


# ============================================================
# 23. SELESAI
# ============================================================

print()
print("=" * 70)
print("SEMUA PROSES SELESAI")
print("=" * 70)

print()

print(
    "Jumlah gambar PRESENT :",
    len(present_files)
)

print(
    "Jumlah gambar ABSENT  :",
    len(absent_files)
)

print(
    "Total data             :",
    total_data
)

print(
    "Prediksi benar         :",
    jumlah_benar
)

print(
    "Prediksi salah         :",
    jumlah_salah
)

print(
    "Akurasi                :",
    f"{akurasi:.2f}%"
)

print()

print("Hasil TXT:")
print(
    os.path.abspath(txt_path)
)

print()

print("Folder comparison:")
print(
    os.path.abspath(COMPARISON_FOLDER)
)

print()

print("Tekan ENTER untuk keluar...")

input()