import streamlit as st
import fitz  # PyMuPDF para trabajar con PDFs
from docx import Document

def leer_metadatos_pdf(archivo_pdf):
    try:
        pdf = fitz.open(archivo_pdf)
        metadatos = pdf.metadata
        pdf.close()
        return metadatos
    except Exception as e:
        return f"Error al leer los metadatos del PDF: {e}"

def modificar_metadatos_pdf(archivo_pdf, archivo_salida, titulo=None, autor=None):
    try:
        pdf = fitz.open(archivo_pdf)
        metadatos = pdf.metadata

        # Modificar los metadatos
        nuevo_metadatos = {}
        if titulo:
            nuevo_metadatos["title"] = titulo
        if autor:
            nuevo_metadatos["author"] = autor

        pdf.set_metadata(nuevo_metadatos)
        pdf.save(archivo_salida)
        pdf.close()
        return "Metadatos modificados correctamente."
    except Exception as e:
        return f"Error al modificar los metadatos del PDF: {e}"

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

def modificar_metadatos_docx(archivo_docx, archivo_salida, titulo=None, autor=None, asunto=None, palabras_clave=None):
    try:
        doc = Document(archivo_docx)
        propiedades = doc.core_properties

        # Modificar los metadatos
        if titulo:
            propiedades.title = titulo
        if autor:
            propiedades.author = autor
        if asunto:
            propiedades.subject = asunto
        if palabras_clave:
            propiedades.keywords = palabras_clave

        # Guardar el documento actualizado
        doc.save(archivo_salida)
        return "Metadatos modificados correctamente."
    except Exception as e:
        return f"Error al modificar los metadatos del DOCX: {e}"

def main():
    st.title("Gestor de Metadatos de Archivos PDF y DOCX")

    archivo_subido = st.file_uploader("Cargar archivo", type=["pdf", "docx"])
    if archivo_subido:
        # Procesar PDF
        if archivo_subido.name.endswith(".pdf"):
            st.subheader("Metadatos del PDF")
            metadatos = leer_metadatos_pdf(archivo_subido)
            if isinstance(metadatos, dict):
                st.json(metadatos)
                st.subheader("Modificar Metadatos")
                titulo = st.text_input("Nuevo Título", value=metadatos.get("title", ""))
                autor = st.text_input("Nuevo Autor", value=metadatos.get("author", ""))
                if st.button("Actualizar Metadatos PDF"):
                    salida_pdf = "archivo_actualizado.pdf"
                    resultado = modificar_metadatos_pdf(archivo_subido, salida_pdf, titulo=titulo, autor=autor)
                    st.success(resultado)
                    st.download_button("Descargar PDF Actualizado", open(salida_pdf, "rb"), file_name=salida_pdf)
            else:
                st.error(metadatos)

        # Procesar DOCX
        elif archivo_subido.name.endswith(".docx"):
            st.subheader("Metadatos del DOCX")
            metadatos = leer_metadatos_docx(archivo_subido)
            if isinstance(metadatos, dict):
                st.json(metadatos)
                st.subheader("Modificar Metadatos")
                titulo = st.text_input("Nuevo Título", value=metadatos.get("title", ""))
                autor = st.text_input("Nuevo Autor", value=metadatos.get("author", ""))
                asunto = st.text_input("Nuevo Asunto", value=metadatos.get("subject", ""))
                palabras_clave = st.text_input("Nuevas Palabras Clave", value=metadatos.get("keywords", ""))
                if st.button("Actualizar Metadatos DOCX"):
                    salida_docx = "archivo_actualizado.docx"
                    resultado = modificar_metadatos_docx(
                        archivo_subido, salida_docx, titulo=titulo, autor=autor, asunto=asunto, palabras_clave=palabras_clave
                    )
                    st.success(resultado)
                    st.download_button("Descargar DOCX Actualizado", open(salida_docx, "rb"), file_name=salida_docx)
            else:
                st.error(metadatos)

if __name__ == "__main__":
    main()
