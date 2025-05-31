import pandas as pd
import numpy as np

# =========================
# FUNGSI FUZZY MEMBERSHIP
# =========================
def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1

# Input Membership Functions
def rendah_r(x): return trapmf(x, 0, 0, 30, 50)
def sedang_r(x): return trimf(x, 40, 60, 80)
def tinggi_r(x): return trapmf(x, 70, 90, 100, 100)

def rendah_w(x): return trapmf(x, 0, 0, 30, 50)
def sedang_w(x): return trimf(x, 40, 60, 80)
def tinggi_w(x): return trapmf(x, 70, 90, 100, 100)

# Output Membership Functions
def buruk(x): return trapmf(x, 0, 0, 30, 50)
def cukup(x): return trimf(x, 40, 60, 80)
def bagus(x): return trapmf(x, 70, 90, 100, 100)

# =========================
# FUZZY MAMDANI INFERENSI
# =========================
output_range = np.linspace(0, 100, 1000)

def mamdani_inferensi(reading, writing):
    μ_r = [rendah_r(reading), sedang_r(reading), tinggi_r(reading)]
    μ_w = [rendah_w(writing), sedang_w(writing), tinggi_w(writing)]

    rules = [
        (μ_r[0], μ_w[0], buruk),
        (μ_r[0], μ_w[1], buruk),
        (μ_r[0], μ_w[2], cukup),
        (μ_r[1], μ_w[0], buruk),
        (μ_r[1], μ_w[1], cukup),
        (μ_r[1], μ_w[2], bagus),
        (μ_r[2], μ_w[0], cukup),
        (μ_r[2], μ_w[1], bagus),
        (μ_r[2], μ_w[2], bagus)
    ]

    output_agg = np.zeros_like(output_range)

    for αr, αw, output_func in rules:
        α = min(αr, αw)
        clipped_output = np.array([min(α, output_func(x)) for x in output_range])
        output_agg = np.maximum(output_agg, clipped_output)

    if output_agg.sum() == 0:
        return 0
    else:
        return np.sum(output_range * output_agg) / np.sum(output_agg)

# =========================
# BACA DATA & PROSES
# =========================
# Ganti path di bawah jika file tidak di folder yang sama
df = pd.read_csv("d:/Kuliah Telkom University/Semester 4/Dasar Kecerdasan Artifisial (DKA)/Tubes/StudentsPerformance.csv")

# Hitung performance score
df['Performance_Score'] = df.apply(
    lambda row: mamdani_inferensi(row['reading score'], row['writing score']), axis=1
)

# Kategorisasi skor
def kategori_performance(score):
    if score <= 50:
        return "buruk"
    elif score <= 75:
        return "cukup"
    else:
        return "bagus"

df['Kategori'] = df['Performance_Score'].apply(kategori_performance)

# Tampilkan 100 data pertama
print("MAMDANI")
print(df[['reading score', 'writing score', 'Performance_Score', 'Kategori']].head(100))



# S U G E N O

# Definisikan ulang fungsi keanggotaan untuk Sugeno (sama seperti Mamdani)
def rendah(x): return trapmf(x, 0, 0, 30, 50)
def sedang(x): return trimf(x, 40, 60, 80)
def tinggi(x): return trapmf(x, 70, 90, 100, 100)

# Definisikan aturan Sugeno dengan output konstan (default logic)
# Output untuk masing-masing rule (z values)
rule_outputs = {
    ('rendah', 'rendah'): 30,
    ('rendah', 'sedang'): 40,
    ('rendah', 'tinggi'): 60,
    ('sedang', 'rendah'): 40,
    ('sedang', 'sedang'): 60,
    ('sedang', 'tinggi'): 80,
    ('tinggi', 'rendah'): 60,
    ('tinggi', 'sedang'): 80,
    ('tinggi', 'tinggi'): 90,
}

# Fungsi Sugeno
def sugeno_inferensi(reading, writing):
    μ_reading = {
        'rendah': rendah(reading),
        'sedang': sedang(reading),
        'tinggi': tinggi(reading)
    }

    μ_writing = {
        'rendah': rendah(writing),
        'sedang': sedang(writing),
        'tinggi': tinggi(writing)
    }

    numerator = 0
    denominator = 0

    for r_key, μ_r in μ_reading.items():
        for w_key, μ_w in μ_writing.items():
            w = min(μ_r, μ_w)
            z = rule_outputs[(r_key, w_key)]
            numerator += w * z
            denominator += w

    return numerator / denominator if denominator != 0 else 0

# Hitung nilai Sugeno untuk setiap data
df['Sugeno_Score'] = df.apply(
    lambda row: sugeno_inferensi(row['reading score'], row['writing score']), axis=1
)

# Kategorisasi hasil Sugeno
df['Sugeno_Kategori'] = df['Sugeno_Score'].apply(kategori_performance)

# Tampilkan 100 data pertama hasil Sugeno
print("SUGENO")
print(df[['reading score', 'writing score', 'Sugeno_Score', 'Sugeno_Kategori']].head(100))

