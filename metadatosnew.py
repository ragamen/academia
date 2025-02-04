import streamlit as st
import fitz  # PyMuPDF para trabajar con PDFs
from docx import Document
import tempfile

def leer_metadatos_pdf(archivo_pdf):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(archivo_pdf.read())  
            temp_pdf_path = temp_pdf.name  
        
        pdf = fitz.open(temp_pdf_path)
        metadatos = pdf.metadata
        pdf.close()
        return metadatos
    except Exception as e:
        return f"Error al leer los metadatos del PDF: {e}"

def modificar_metadatos_pdf(archivo_pdf, titulo=None, autor=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(archivo_pdf.read())  
            temp_pdf_path = temp_pdf.name  
        
        pdf = fitz.open(temp_pdf_path)
        nuevo_metadatos = {}

        if titulo:
            nuevo_metadatos["title"] = titulo
        if autor:
            nuevo_metadatos["author"] = autor

        pdf.set_metadata(nuevo_metadatos)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as salida_pdf:
            pdf.save(salida_pdf.name)
            pdf.close()
            return salida_pdf.name, "Metadatos modificados correctamente."
    except Exception as e:
        return None, f"Error al modificar los metadatos del PDF: {e}"

def leer_metadatos_docx(archivo_docx):
    try:
        doc = Document(archivo_docx)
        propiedades = doc.core_properties
        return {
            "title": propiedades.title,
            "author": propiedades.author,
            "subject": propiedades.subject,
            "keywords": propiedades.keywords,
            "comments": propiedades.comments,
        }
    except Exception as e:
        return f"Error al leer los metadatos del DOCX: {e}"

def modificar_metadatos_docx(archivo_docx, titulo=None, autor=None, asunto=None, palabras_clave=None):
    try:
        doc = Document(archivo_docx)
        propiedades = doc.core_properties

        if titulo:
            propiedades.title = titulo
        if autor:
            propiedades.author = autor
        if asunto:
            propiedades.subject = asunto
        if palabras_clave:
            propiedades.keywords = palabras_clave

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as salida_docx:
            doc.save(salida_docx.name)
            return salida_docx.name, "Metadatos modificados correctamente."
    except Exception as e:
        return None, f"Error al modificar los metadatos del DOCX: {e}"

def main():
    st.title("Gestor de Metadatos de Archivos PDF y DOCX")

    archivo_subido = st.file_uploader("Cargar archivo", type=["pdf", "docx"])
    if archivo_subido:
        # Procesar PDF
        if archivo_subido.name.endswith(".pdf"):
            st.subheader("Metadatos del PDF")
            archivo_subido.seek(0)  # Reiniciar el puntero del archivo
            metadatos = leer_metadatos_pdf(archivo_subido)
            if isinstance(metadatos, dict):
                st.json(metadatos)
                st.subheader("Modificar Metadatos")
                titulo = st.text_input("Nuevo Título", value=metadatos.get("title", ""))
                autor = st.text_input("Nuevo Autor", value=metadatos.get("author", ""))
                if st.button("Actualizar Metadatos PDF"):
                    archivo_subido.seek(0)  # Reiniciar el archivo antes de volver a leerlo
                    archivo_salida, resultado = modificar_metadatos_pdf(archivo_subido, titulo=titulo, autor=autor)
                    if archivo_salida:
                        st.success(resultado)
                        st.download_button("Descargar PDF Actualizado", open(archivo_salida, "rb"), file_name="archivo_actualizado.pdf")
                    else:
                        st.error(resultado)
            else:
                st.error(metadatos)

        # Procesar DOCX
        elif archivo_subido.name.endswith(".docx"):
            st.subheader("Metadatos del DOCX")
            archivo_subido.seek(0)  # Reiniciar el puntero del archivo
            metadatos = leer_metadatos_docx(archivo_subido)
            if isinstance(metadatos, dict):
                st.json(metadatos)
                st.subheader("Modificar Metadatos")
                titulo = st.text_input("Nuevo Título", value=metadatos.get("title", ""))
                autor = st.text_input("Nuevo Autor", value=metadatos.get("author", ""))
                asunto = st.text_input("Nuevo Asunto", value=metadatos.get("subject", ""))
                palabras_clave = st.text_input("Nuevas Palabras Clave", value=metadatos.get("keywords", ""))
                if st.button("Actualizar Metadatos DOCX"):
                    archivo_subido.seek(0)  # Reiniciar el archivo antes de volver a leerlo
                    archivo_salida, resultado = modificar_metadatos_docx(
                        archivo_subido, titulo=titulo, autor=autor, asunto=asunto, palabras_clave=palabras_clave
                    )
                    if archivo_salida:
                        st.success(resultado)
                        st.download_button("Descargar DOCX Actualizado", open(archivo_salida, "rb"), file_name="archivo_actualizado.docx")
                    else:
                        st.error(resultado)
            else:
                st.error(metadatos)

if __name__ == "__main__":
    main()
