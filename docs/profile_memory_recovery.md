cd ~/AgenticAI_LangGraph && cat > docs/profile_memory_recovery.md <<'EOF'
# Profile Memory: Root Cause Analysis & Database Recovery

## 1. Overview

Dokumen ini mencatat investigasi dan pemulihan data `profile_memory.sqlite`
pada project AgenticAI LangGraph.

Profile memory digunakan untuk menyimpan informasi jangka panjang pengguna,
termasuk nama, bahasa pilihan, dan gaya respons.

Database menggunakan SQLite dengan tabel `user_profiles`.

## 2. Incident Summary

| Item | Detail |
| --- | --- |
| Component | Profile Memory |
| Database | `profile_memory.sqlite` |
| User ID | `local-user` |
| Environment | Windows 11 dan Ubuntu Linux |
| Symptom | API `GET /profile` mengembalikan `{}` |
| HTTP Status | 200 OK |
| Final Status | Recovered and verified |

Gejala awal menunjukkan endpoint API berhasil diakses, tetapi tidak menemukan
record profile untuk `local-user`.

## 3. Root Cause Analysis

### 3.1 Temuan utama

Project berjalan pada dua environment terpisah:

- Windows 11: `D:\Project\AgenticAI_LangGraph`
- Ubuntu: `/home/fajar/AgenticAI_LangGraph`

Kedua environment memiliki file SQLite lokal masing-masing. File database
tidak otomatis tersinkronisasi antara Windows dan Ubuntu.

### 3.2 Bukti investigasi

Database Ubuntu sebelum recovery:

- File ditemukan di `/home/fajar/AgenticAI_LangGraph/profile_memory.sqlite`.
- Ukuran file 12 KB.
- Tabel `user_profiles` memiliki 0 record.

Database Windows:

- File ditemukan di `D:\Project\AgenticAI_LangGraph\profile_memory.sqlite`.
- Ukuran file 12 KB.
- Memiliki 1 record untuk `local-user`.

Record yang ditemukan pada database Windows:

| Field | Value |
| --- | --- |
| `user_id` | `local-user` |
| `name` | `Fajar` |
| `preferred_language` | `English` |
| `response_style` | `concise` |
| `updated_at` | `2026-10-25T10:17:41.974031+00:00` |

### 3.3 Kesimpulan

Penyebab langsung profile tidak ditemukan pada API Ubuntu adalah database
yang digunakan environment Ubuntu tidak memiliki record `local-user`.

Database Windows masih memiliki record tersebut. Investigasi menunjukkan
perbedaan data lokal antar-environment, bukan bukti adanya kerusakan kode
profile memory.

## 4. Database Recovery Procedure

Recovery dilakukan dengan memindahkan database Windows ke environment Ubuntu
melalui SCP.

### Step 1 — Stop API server

Uvicorn dihentikan menggunakan `Ctrl+C` sebelum database aktif diganti.

### Step 2 — Backup database Ubuntu

```bash
cp -a profile_memory.sqlite profile_memory.sqlite.before_restore_20261116.bak
```

Backup database Ubuntu dipertahankan sebagai salinan sebelum recovery.

### Step 3 — Copy database dari Windows
Di Windows PowerShell:

```powershell
Copy-Item "D:\Project\AgenticAI_LangGraph\profile_memory.sqlite" "$env:USERPROFILE\Downloads\profile_memory.sqlite"
```

### Step 4 — Transfer menggunakan SCP
Di Windows PowerShell:

```powershell
scp "$env:USERPROFILE\Downloads\profile_memory.sqlite" fajar@IP_SERVER:/home/fajar/AgenticAI_LangGraph/profile_memory_windows.sqlite
```

Ganti IP_SERVER dengan alamat server Ubuntu yang digunakan.

### Step 5 — Verify transferred database
Database hasil transfer diperiksa menggunakan SQLite dalam mode read-only.

Hasil verifikasi menunjukkan:
- Total records: 1
- User ID: local-user
- Name: Fajar
- Preferred language: English
- Response style: concise

### Step 6 — Restore database
Setelah transfer berhasil diverifikasi:

```bash
mv profile_memory_windows.sqlite profile_memory.sqlite
```

Database Ubuntu sebelum recovery tetap tersedia sebagai:
`profile_memory.sqlite.before_restore_20261116.bak`


## 5. Recovery Validation

### 5.1 Direct database validation
Pemanggilan profile_memory.get_profile("local-user") berhasil
mengembalikan record profile.

### 5.2 API validation
Endpoint:
```
GET /profile?user_id=local-user
```

Mengembalikan HTTP 200 dengan data:
```
{
  "user_id": "local-user",
  "name": "Fajar",
  "preferred_language": "English",
  "response_style": "concise",
  "updated_at": "2026-10-25T10:17:41.974031+00:00"
}
```


### 5.3 LangGraph integration validation
Pengujian dilakukan melalui endpoint:
`POST /chat`


Pertanyaan:
`What is my name and what response style do I prefer?`


Jawaban agent:
`Your name is Fajar, and you prefer a concise response style.`


Hasil ini membuktikan profile berhasil dibaca dan digunakan dalam
pemrosesan percakapan.

## 6. Regression Testing
Regression test dijalankan menggunakan:
```bash
pytest -v
```

Hasil:
```
100 passed, 4 deselected
```

Tidak ada test yang gagal. Empat test tidak dijalankan karena konfigurasi
seleksi test yang berlaku.

## 7. Prevention and Recommendations
1. Perlakukan database SQLite pada Windows dan Ubuntu sebagai data lokal
   yang terpisah.
2. Pastikan environment menggunakan database path yang benar.
3. Buat backup sebelum melakukan restore atau mengganti database.
4. Verifikasi isi database hasil transfer sebelum menjadikannya database aktif.
5. Jangan memasukkan database runtime atau backup ke Git tanpa keputusan
   eksplisit mengenai kebutuhan dan keamanan penyimpanan datanya.
6. Untuk penggunaan jangka panjang atau multi-instance, pertimbangkan
   mekanisme backup dan sinkronisasi database yang terencana.
## 8. Final Status

- [x] Root cause investigated.
- [x] Windows database identified as containing the expected profile.
- [x] Ubuntu database backed up.
- [x] Database transferred and verified.
- [x] Ubuntu database restored.
- [x] Direct profile retrieval verified.
- [x] API profile endpoint verified.
- [x] LangGraph profile integration verified.
- [x] Regression tests passed.
**Recovery status: SUCCESS**
