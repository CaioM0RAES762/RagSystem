scripts/python/
  rag/
    __init__.py
    rag_system.py              # facade + classe pública RAGSystem
    config.py                  # env, defaults, modelos, paths
    logger.py                  # _log_step, _log_info...
    page_selection.py          # parse_page_selection, normalize_page_num
    preflight.py               # preflight_pdf_pages, PagePreflight, summary
    ocr.py                     # run_ocr_if_needed
    render.py                  # render_page_to_png, useful check, phash dedupe
    text_extract.py            # extrair texto por página / legacy
    translate.py               # detectar idioma e traduzir (por página)
    chunking.py                # dividir chunks / paginas_para_chunks_texto
    chroma_store.py            # add/get/query no Chroma (texto+imagem)
    pg_store.py                # _pg_conn, upsert manual, salvar imagem
    vision.py                  # prompt + gerar descrição de imagem
    solver.py                  # gerar solução com/sem imagens (LLM)
  rag_pdf_processor.py         # arquivo compatível com imports antigos
