import os
import base64
import json
import hashlib
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PaginatedPipelineOptions
from docling.chunking import HybridChunker
from clip import classify_image_graph_or_not, describe_image, describe_slides, init_clip_model

def get_converter(describe_images=True):
    pdf_pipeline_options = PdfPipelineOptions()
    docx_pipeline_options = PaginatedPipelineOptions()
    pdf_pipeline_options.generate_picture_images = describe_images
    docx_pipeline_options.generate_picture_images = describe_images

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_pipeline_options)}
    )

def pptx_to_images(source_filepath: str) -> list[bytes]:
    import comtypes.client
    import tempfile
    from pathlib import Path
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1 # Set to 0 to suppress PowerPoint window
    images = []
    try:
        # COM requires absolute paths
        abs_source_path = str(Path(source_filepath).resolve())
        presentation = powerpoint.Presentations.Open(abs_source_path)
        export_filter_name = "PNG"
        # Create a self-cleaning temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, slide in enumerate(presentation.Slides):
                image_filepath = temp_path / f"slide_{i+1}.png"
                # output_path = Path(f"slides_spire/slide_{i+1}.png")
                # abs_output_path = str(output_path.resolve())
                slide.Export(str(image_filepath), export_filter_name, 1920, 1080)
                # slide.Export(abs_output_path, export_filter_name,1920, 1080)
                with open(image_filepath, "rb") as f:
                    images.append(f.read())
                print(f"Processed slide {i+1} into memory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'presentation' in locals() and presentation:
            presentation.Close()
        if 'powerpoint' in locals() and powerpoint:
            powerpoint.Quit()
            del powerpoint
            print("PowerPoint application closed.")
    return images

def generate_chunks_from_pptx(source_filepath, output_path, vlm_model):
    filename = os.path.splitext(os.path.basename(source_filepath))[0]
    print(f"Converting PPTX to images...")
    slide_images = pptx_to_images(source_filepath)
    print(f"Images converted successfully")
    chunks = []
    for i,image in enumerate(slide_images):
        # Generate slide image descrpiptions(NO classification needed)
        try:
            description = describe_slides(image, vlm_model)
            chunks.append({
                "id": hashlib.sha256(description.encode('utf-8')).hexdigest(),
                "source_filename": filename,
                "page_number": i+1,
                "content": description
            })
            print(f"    -> Description generated successfully")
        except Exception as e:
            print(f"Error describing slide{i+1}: {e}")
    # Generate chunks with slide image descriptions and store it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(f'{output_path}/{filename}.jsonl', 'w', encoding='utf-8') as f:
        for line in chunks:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return f'{output_path}/{filename}.jsonl'

# Generate chunks for vector RAG
def process_document_to_chunks(source_filepath, output_path, converter, tokenizer, clip_model, clip_processor, device, vlm_model=None, describe_images=True):
    doc = converter.convert(source_filepath).document
    filename = os.path.splitext(os.path.basename(source_filepath))[0]
    if filename.endswith('.pptx'):
        return generate_chunks_from_pptx(source_filepath, output_path, vlm_model)
    if describe_images:
        pl = {}
        for picture in doc.pictures:
            pl[picture.self_ref] = picture.image.uri.path.split(',')[1]
        jsonl_lines = []
        # Process text chunks with pictures
        chunker = HybridChunker(tokenizer=tokenizer)
        chunks = list(chunker.chunk(doc))
        for chunk in chunks:
            pics = []
            # Collect all pictures in this chunk
            for item in chunk.meta.doc_items:
                # Check for pictures
                if item.parent and str(item.parent.cref).startswith('#/pictures/'): pics.append(item.parent.cref)
                elif str(item.self_ref).startswith('#/pictures/'): pics.append(item.self_ref)
            # Process pictures - get first graph/diagram description if available
            pic_descs = []
            for pic_ref in pics:
                print(f"  Processing picture: {pic_ref}")
                if pic_ref in pl:
                    img_bytes_b64 = pl[pic_ref]
                    img_bytes = base64.b64decode(img_bytes_b64)
                    if classify_image_graph_or_not(clip_model, clip_processor, device, img_bytes):
                        try:
                            pic_desc = describe_image(img_bytes, vlm_model)
                            pic_descs.append(pic_desc)
                            print(f"    -> Description generated successfully")
                        except Exception as e:
                            print(f"    -> Error generating description: {e}")
                    else:
                        print(f"    -> Image is not a graph or diagram")
                else: print(f"    -> No image found for {pic_ref}")
            heading = f"{' > '.join(str(h) for h in chunk.meta.headings if h)}\n" if chunk.meta.headings else ''
            content = heading + chunk.text
            jsonl_lines.append({
                "id": hashlib.sha256(content[:100].encode('utf-8')).hexdigest(),
                "source_filename": filename,
                "page_number": chunk.meta.doc_items[0].prov[0].page_no,
                "content": content,
                "picture_description": pic_descs
            })
    else:
        jsonl_lines = []
        # Process text chunks without pictures
        chunker = HybridChunker(tokenizer=tokenizer)
        chunks = list(chunker.chunk(doc))
        for chunk in chunks:
            heading = f"{' > '.join(str(h) for h in chunk.meta.headings if h)}\n" if chunk.meta.headings else ''
            content = heading + chunk.text
            jsonl_lines.append({
                "id": hashlib.sha256(content[:100].encode('utf-8')).hexdigest(),
                "source_filename": filename,
                "page_number": chunk.meta.doc_items[0].prov[0].page_no,
                "content": content,
                "picture_description": []
            })
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(f'{output_path}/{filename}.jsonl', 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return f'{output_path}/{filename}.jsonl'

if __name__ == '__main__':
    converter = get_converter()
    clip_model, clip_processor, device = init_clip_model()
    output_path = process_document_to_chunks('aws.pptx','markdown', converter, clip_model, clip_processor, device)
    print(f'Markdown file generated: {output_path}')