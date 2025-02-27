with open("cfg/config.yaml", "rb") as f:
    content = f.read()

for i, byte in enumerate(content):
    if byte > 127:  # Karakter non-ASCII
        print(f"Karakter tidak valid di posisi {i}: {hex(byte)}")
