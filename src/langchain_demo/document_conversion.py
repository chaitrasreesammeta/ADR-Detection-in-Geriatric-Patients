from config import DATA_DIR
import pymupdf4llm
from pathlib import Path

stopp_start_file = DATA_DIR / "STOPP-START v3 .pdf"
output_markdown_file = DATA_DIR / "STOPP-START v3 .md"


def run(
    input_file: Path,
    output_file: Path,
    page_start_number: int,
    page_end_number: int,
):
    kwargs = {
        "ignore_code": True,
        "ignore_graphics": True,
        "ignore_images": True,
        "header": False,
        "footer": False,
        "pages": range(page_start_number, page_end_number + 1),
    }

    markdown_text = pymupdf4llm.to_markdown(input_file, **kwargs)
    output_file.write_text(data=markdown_text, encoding="utf-8")
    print(f"{input_file.name} converted to {output_file.name}")


if __name__ == "__main__":
    run(
        input_file=stopp_start_file,
        output_file=output_markdown_file,
        page_start_number=0,
        page_end_number=15,
    )