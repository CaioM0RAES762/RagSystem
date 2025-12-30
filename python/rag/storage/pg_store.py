# rag/storage/pg_store.py

import json
from typing import Optional, Dict, Any, List

import psycopg2
from psycopg2.extras import RealDictCursor

from rag.logger import log_ok as _log_ok, log_warn as _log_warn, log_info as _log_info


class PGStore:
    """
    Camada de acesso PostgreSQL SAFE:
    - Conexão lazy e cache
    - Schema discovery (embedding_curto vector / json)
    - Upsert manual e imagens
    """

    def __init__(
        self,
        *,
        host: str,
        dbname: str,
        user: str,
        password: str,
        port: int = 5432,
    ):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port

        self._conn_cache = None
        self._schema_cache = None

    # --------------------------
    # Conexão
    # --------------------------
    def connect(self):
        if self._conn_cache is not None:
            return self._conn_cache

        if not all([self.host, self.dbname, self.user, self.password]):
            _log_info("[PGStore] Variáveis de banco não configuradas.")
            self._conn_cache = None
            return None

        try:
            conn = psycopg2.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            conn.autocommit = True
            self._conn_cache = conn
            _log_ok("[PGStore] Conectado ao PostgreSQL.")
            return conn
        except Exception as e:
            _log_warn(f"[PGStore] Falha ao conectar: {e}")
            self._conn_cache = None
            return None

    def close(self):
        try:
            if self._conn_cache:
                self._conn_cache.close()
                _log_ok("[PGStore] Conexão encerrada.")
        except Exception:
            pass
        self._conn_cache = None

    # --------------------------
    # Schema cache
    # --------------------------
    def schema(self, conn) -> Dict[str, Any]:
        if self._schema_cache is not None:
            return self._schema_cache

        schema = {"manual_imagens_cols": set(), "has_pgvector": False}

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_name='manual_imagens';
            """)
            rows = cur.fetchall()
            cur.close()

            cols = set()
            for col, udt in rows:
                cols.add(col)
                if udt == "vector":
                    schema["has_pgvector"] = True

            schema["manual_imagens_cols"] = cols

        except Exception as e:
            _log_warn(f"[PGStore] Falha ao detectar schema: {e}")

        self._schema_cache = schema
        return schema

    # --------------------------
    # Upsert Manual
    # --------------------------
    def upsert_manual(self, conn, manual_id: int, nome_maquina: Optional[str], pdf_path: Optional[str]):
        if manual_id is None:
            return

        cur = None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("SELECT id FROM manuais WHERE id = %s;", (manual_id,))
            row = cur.fetchone()
            if row:
                return

            nome_maquina_final = nome_maquina or f"Máquina {manual_id}"
            nome_manual_final = nome_maquina or f"Manual ID {manual_id}"
            caminho_pdf_final = pdf_path or ""

            cur.execute("""
                INSERT INTO manuais (
                    id, maquina_id, nome_maquina, nome_manual, caminho_pdf, status
                )
                VALUES (%s, %s, %s, %s, %s, 'processado')
                ON CONFLICT (id) DO NOTHING;
            """, (manual_id, manual_id, nome_maquina_final, nome_manual_final, caminho_pdf_final))

            _log_ok(f"[PGStore] manuais upsert ok id={manual_id}")

        except Exception as e:
            _log_warn(f"[PGStore] upsert_manual falhou id={manual_id}: {e}")
        finally:
            if cur:
                cur.close()

    # --------------------------
    # Upsert Imagem
    # --------------------------
    def upsert_manual_imagem(
        self,
        conn,
        *,
        manual_id: int,
        pagina: int,
        indice: int,
        caminho: str,
        hash_md5: str,
        descricao_completa: str,
        descricao_curta: str,
        modelo_vision: str,
    ) -> Optional[int]:

        cur = None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT id
                FROM manual_imagens
                WHERE manual_id = %s AND pagina = %s AND indice_imagem = %s;
            """, (manual_id, pagina, indice))
            row = cur.fetchone()

            if row:
                cur.execute("""
                    UPDATE manual_imagens
                    SET arquivo_imagem=%s, hash_md5=%s,
                        descricao_completa=%s, descricao_curta=%s,
                        modelo_vision=%s,
                        foi_reprocessada=TRUE,
                        atualizado_em=NOW()
                    WHERE id=%s
                    RETURNING id;
                """, (caminho, hash_md5, descricao_completa, descricao_curta, modelo_vision, row["id"]))
                rid = cur.fetchone()["id"]
                _log_ok(f"[PGStore] manual_imagens atualizado id={rid}")
                return rid

            cur.execute("""
                INSERT INTO manual_imagens (
                    manual_id, pagina, indice_imagem,
                    arquivo_imagem, hash_md5,
                    descricao_completa, descricao_curta,
                    modelo_vision, foi_reprocessada
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                RETURNING id;
            """, (manual_id, pagina, indice, caminho, hash_md5, descricao_completa, descricao_curta, modelo_vision))

            rid = cur.fetchone()["id"]
            _log_ok(f"[PGStore] manual_imagens criado id={rid}")
            return rid

        except Exception as e:
            _log_warn(
                f"[PGStore] upsert_manual_imagem falhou manual={manual_id} pag={pagina}: {e}")
            return None
        finally:
            if cur:
                cur.close()

    # --------------------------
    # Salvar embedding curto (vector / json)
    # --------------------------
    def save_embedding_curto(self, conn, row_id: int, embedding: List[float]):
        if not conn or not row_id or not embedding:
            return

        schema = self.schema(conn)
        cols = schema.get("manual_imagens_cols", set())

        has_vector_col = "embedding_curto" in cols
        has_json_col = "embedding_curto_json" in cols

        cur = None
        try:
            cur = conn.cursor()

            if has_vector_col:
                vec_str = "[" + ",".join([f"{x:.8f}" for x in embedding]) + "]"
                cur.execute(
                    "UPDATE manual_imagens SET embedding_curto = %s WHERE id = %s;", (vec_str, row_id))
                return

            if has_json_col:
                cur.execute(
                    "UPDATE manual_imagens SET embedding_curto_json = %s WHERE id = %s;",
                    (json.dumps(embedding), row_id),
                )
                return

        except Exception as e:
            _log_warn(
                f"[PGStore] save_embedding_curto falhou id={row_id}: {e}")
        finally:
            if cur:
                cur.close()
