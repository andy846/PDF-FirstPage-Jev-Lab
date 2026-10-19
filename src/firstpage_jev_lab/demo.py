"""Completely synthetic PDF fixture. Contains no customer/production data."""
from pathlib import Path
import fitz


def make_demo(path: str | Path, lengths=(3, 2, 4, 3, 2, 3, 3)) -> tuple[Path, list[int]]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    starts = []
    current = 1
    with fitz.open() as doc:
        for letter_index, count in enumerate(lengths, 1):
            starts.append(current)
            for letter_page in range(1, count+1):
                p = doc.new_page(width=595, height=842)
                if letter_page == 1:
                    p.insert_text((50, 60), "MEMBER STATEMENT")
                    p.insert_text((50, 88), "Statement Date: [SYNTHETIC]")
                else:
                    p.insert_text((50, 60), "CONTINUATION OF STATEMENT")
                    # Simulate a misleading marker that also occurs on continuation pages.
                    if letter_index == 2 and letter_page == 2:
                        p.insert_text((50, 88), "Statement Date: [SYNTHETIC]")
                p.insert_text((450, 62), f"Page {letter_page} of {count}")
                p.insert_text((50, 140), "Synthetic demonstration data only. No customer information.")
                current += 1
        doc.save(out)
    return out, starts
