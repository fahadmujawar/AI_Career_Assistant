import fitz


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in document:
        text += page.get_text("text")
        text += "\n"

    page_count = len(document)

    document.close()

    uploaded_file.seek(0)

    return text, page_count