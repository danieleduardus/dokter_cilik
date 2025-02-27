# Buka dan baca file YAML dengan mode binary
with open("cfg/config.yaml", "rb") as file:
    content = file.read()

# Hapus karakter non-ASCII
clean_content = content.decode("utf-8", errors="ignore")

# Simpan kembali file tanpa karakter yang menyebabkan error
with open("cfg/config.yaml", "w", encoding="utf-8") as file:
    file.write(clean_content)

print("File config.yaml berhasil dibersihkan!")
