
"""
Aplicació Streamlit: resum automàtic de textos amb FLAN-T5.

Execució local:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

# Seleccionem GPU si està disponible; en cas contrari, CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def carregar_model():
    """
    Carrega el tokenitzador i el model una sola vegada.
    Així evitem descarregar els pesos a cada petició de l'usuari.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    return tokenizer, model


def resumir(text, tokenizer, model):
    """Rep un text i retorna'n el resum generat pel model."""

    if not text or not text.strip():
        return "Introdueix un text abans de demanar-ne el resum."

    # FLAN-T5 necessita una instrucció que indiqui la tasca.
    prompt = "summarize: " + text.strip()

    # Convertim el text en tensors i limitem la longitud d'entrada.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Generem la resposta sense calcular gradients:
    # només fem inferència, no entrenament.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )

    # Convertim els tokens generats en una cadena de text llegible.
    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


def main():
    """Construeix la interfície web de l'aplicació amb Streamlit."""

    st.set_page_config(
        page_title="Resum automàtic de textos",
        page_icon="📝",
        layout="centered"
    )

    st.title("Resum automàtic de textos")

    st.write(
        "Aplicació basada en el model google/flan-t5-base."
    )

    # Carreguem el tokenitzador i el model una sola vegada.
    with st.spinner("Carregant el model..."):
        tokenizer, model = carregar_model()

    # Quadre de text d'entrada on l'usuari enganxa el contingut.
    text_entrada = st.text_area(
        "Text que vols resumir",
        height=250,
        placeholder="Enganxa aquí el text..."
    )

    # Botó per iniciar la generació del resum.
    if st.button("Generar resum", type="primary"):

        if not text_entrada or not text_entrada.strip():

            st.warning(
                "Introdueix un text abans de demanar-ne el resum."
            )

        else:

            try:
                # Mostrem un indicador mentre es genera el resum.
                with st.spinner("Generant el resum..."):

                    resultat = resumir(
                        text_entrada,
                        tokenizer,
                        model
                    )

                # Mostrem el resultat generat pel model.
                st.subheader("Resum generat")

                st.text_area(
                    "Resultat",
                    value=resultat,
                    height=180
                )

            except Exception as error:

                st.error(
                    f"No s'ha pogut generar el resum: {error}"
                )


if __name__ == "__main__":
    main()
