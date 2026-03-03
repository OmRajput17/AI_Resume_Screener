import docx
import sys

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        with open('guide_v2.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_text))
        print("Success")
    except Exception as e:
        print(f"Error reading docx: {e}")

if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'ResumeScreener_v2_Fixed.docx'
    read_docx(file_path)
