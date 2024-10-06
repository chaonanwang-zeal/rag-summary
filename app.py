import streamlit as st
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA


def process_pdf(uploaded_pdf, api_key):
    try:
        CHUNK_SIZE = 700
        CHUNK_OVERLAP = 100

        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())
            temp_pdf_path = temp_pdf.name

        pdf_loader = PyPDFLoader(temp_pdf_path)
        split_pdf_document = pdf_loader.load_and_split()

        st.write(f"PDF Loaded: {len(split_pdf_document)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        context = "\n\n".join(str(p.page_content) for p in split_pdf_document)
        texts = text_splitter.split_text(context)

        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            google_api_key=api_key
        )

        vector_index = FAISS.from_texts(texts, embeddings)
        retriever = vector_index.as_retriever(search_kwargs={"k": 5})

        return retriever
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


def process_vtt_file(vtt_content):
    try:
        lines = vtt_content.splitlines()
        filtered_lines = [line for line in lines if line.startswith('<v')]
        return '\n'.join(filtered_lines)
    except Exception as e:
        st.error(f"Error processing VTT: {str(e)}")
        return ""


def summarize_vtt(vtt_content, retriever, api_key, use_pdf):
    try:
        gemini_model = ChatGoogleGenerativeAI(
            model='gemini-1.5-pro-latest',
            google_api_key=api_key,
            temperature=0.8
        )

        qa_chain = RetrievalQA.from_chain_type(
            gemini_model,
            retriever=retriever if use_pdf else None,  # 如果没有PDF，retriever将为None
            return_source_documents=True
        )

        if use_pdf:
            question = f"""
            PDFの内容を基づいて、VTTファイルの内容を会議記録としてまとめてください。出力フォーマットは以下のようにしてください。

            1. 参加者: （会議に参加した人たちの名前）
            2. 会議日時: （会議が行われた日時）
            3. 議事録:
                - (議論された主な内容を7つの箇条書きで示してください)
            4. 結論:
                - (得られた結論)
            5. 根拠:
                - (結論に至った理由やPDFの内容に基づいた根拠)

            VTTファイルの内容はこちらです:

            {vtt_content}
            """
        else:
            question = f"""
            VTTファイルの内容を、会議記録としてまとめてください。出力フォーマットは以下のようにしてください。

            1. 参加者: （会議に参加した人たちの名前）
            2. 会議日時: （会議が行われた日時）
            3. 議事録:
                - (議論された主な内容を7つの箇条書きで示してください)
            4. 結論:
                - (得られた結論)

            VTTファイルの内容はこちらです:

            {vtt_content}
            """

        result = qa_chain.invoke({"query": question})

        return result["result"]
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""


def main():
    st.title("⚡RAG＋VTT議事録BOT🤖")
    st.markdown(
        """
    手順：
    1. GoogleのAPIキーを入力する
    2. (オプション)RAG対象のPDFをアップロードする(私はPDFしか読まない!)
    3. (必須)サマリ対象のVTTファイルをアップロードする(Teamsから字幕ファイルを出力されたフィアル)
    4. 結果待つ
    """
    )

    api_key = st.text_input("Enter your Google API key", type="password")

    uploaded_pdf = st.file_uploader(
        "Upload your PDF file (optional)", type="pdf")
    uploaded_vtt = st.file_uploader(
        "Upload your VTT file (required)", type="vtt")

    if st.button("Summarize"):
        if not uploaded_vtt:
            st.error("VTTファイルは必須です。VTTファイルをアップロードしてください。")
        else:

            vtt_content = uploaded_vtt.read().decode('utf-8')
            processed_vtt = process_vtt_file(vtt_content)

            if uploaded_pdf:
                with st.spinner("Processing PDF..."):
                    retriever = process_pdf(uploaded_pdf, api_key)
                    if retriever:
                        with st.spinner("Summarizing VTT based on PDF..."):
                            summary = summarize_vtt(
                                processed_vtt, retriever, api_key, use_pdf=True
                            )
                            st.markdown(f"```text{summary}```")
            else:
                with st.spinner("Summarizing VTT without PDF..."):
                    summary = summarize_vtt(
                        processed_vtt, None, api_key, use_pdf=False
                    )
                    st.markdown(f"```text{summary}```")


if __name__ == "__main__":
    main()
