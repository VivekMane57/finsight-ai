import fitz

class DocumentProcessor:

    @staticmethod
    def extract_pdf_text(file_path: str):

        document = fitz.open(file_path)

        full_text = ""

        for page in document:
            full_text += page.get_text()

        document.close()

        return full_text