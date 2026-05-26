Folder ini disediakan untuk mengikuti struktur tugas.

Workflow aktif tetap berada di folder `.github/workflows`.

Untuk sinkronisasi otomatis versi dokumentasi workflow ke folder `.workflow/workflows`, jalankan:

```powershell
powershell -ExecutionPolicy Bypass -File .workflow/sync-workflow-docs.ps1
```

Script akan:
- Menyalin semua file `.yml`/`.yaml` dari `.github/workflows` ke `.workflow/workflows`
- Menghapus file dokumentasi lama yang sudah tidak ada di sumber
