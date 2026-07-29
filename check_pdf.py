import fitz

doc = fitz.open("data/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025 (1).pdf")
print("Total pages:", len(doc))

for i in range(min(10, len(doc))):
    print(f"--- Page {i} ---")
    print(doc[i].get_text()[:150])
    print()