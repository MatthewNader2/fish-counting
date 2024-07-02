import docx
import fitz  # PyMuPDF

def extract_images_from_docx(docx_path, output_dir):
    doc = docx.Document(docx_path)
    for i, rel in enumerate(doc.part.rels):
        if "image" in doc.part.rels[rel].target_ref:
            img = doc.part.rels[rel].target_part.blob
            with open(f"{output_dir}/image_{i+1}.png", "wb") as f:
                f.write(img)

def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        for img_num, img in enumerate(doc.get_page_images(page_num)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            with open(f"{output_dir}/image_{page_num+1}_{img_num+1}.png", "wb") as f:
                f.write(image_bytes)

# Extract images from docx files
extract_images_from_docx("docs/program.docx", "data")
extract_images_from_docx("docs/graphs.docx", "data")
extract_images_from_docx("docs/Challenge2.docx", "data")
extract_images_from_docx("docs/test.docx", "data")
extract_images_from_docx("docs/exp.docx", "data")

# Extract images from pdf files
extract_images_from_pdf("data/counting_fish_2.pdf", "data")
