BEGIN;

LOCK TABLE chunks IN ACCESS EXCLUSIVE MODE;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM chunks WHERE embedding IS NOT NULL) THEN
        RAISE EXCEPTION
            'Migration aborted: chunks already contain embeddings';
    END IF;
END $$;

ALTER TABLE chunks
    ALTER COLUMN embedding TYPE vector(384)
    USING NULL::vector(384);

ALTER TABLE chunks
    ADD COLUMN embedding_model TEXT,
    ADD COLUMN embedded_at TIMESTAMPTZ;

COMMIT;
