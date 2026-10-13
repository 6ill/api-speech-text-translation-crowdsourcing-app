import json
import random

ID_FILE = "OpenSubtitles.en-id.id"
EN_FILE = "OpenSubtitles.en-id.en"
OUTPUT_FILE = "mt_test_set.jsonl"
SAMPLE_SIZE = 1000

def prepare_test_set():
    print("Membaca file dataset...")
    with open(ID_FILE, 'r', encoding='utf-8') as f_id, \
         open(EN_FILE, 'r', encoding='utf-8') as f_en:
        
        id_lines = f_id.readlines()
        en_lines = f_en.readlines()

    # Validasi: Jumlah baris harus sama persis
    if len(id_lines) != len(en_lines):
        raise ValueError("Jumlah baris file .id dan .en tidak sama!")

    print(f"Total baris ditemukan: {len(id_lines)}")
    
    # Gabungkan baris dan filter data kotor (string kosong)
    valid_pairs = []
    for i in range(len(id_lines)):
        src = id_lines[i].strip()
        tgt = en_lines[i].strip()
        
        # Filter sederhana: Abaikan baris kosong atau terlalu pendek (misal cuma isi "Oh.")
        if len(src) > 3 and len(tgt) > 3:
            valid_pairs.append({
                "id": str(i), 
                "source_text": src, 
                "target_text": tgt
            })

    print(f"Total baris valid: {len(valid_pairs)}")

    # Ambil sampel acak
    print(f"Mengambil {SAMPLE_SIZE} sampel acak...")
    sampled_data = random.sample(valid_pairs, min(SAMPLE_SIZE, len(valid_pairs)))

    # Tulis ke file JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in sampled_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Selesai! File {OUTPUT_FILE} berhasil dibuat.")

if __name__ == "__main__":
    prepare_test_set()