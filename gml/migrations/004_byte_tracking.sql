-- Migration 004 — byte-tracking trigger.
--
-- Maintains users.bytes_used automatically as memories are inserted,
-- updated, or deleted. The application MUST NOT write to bytes_used
-- directly — read-only quota status is via the `user_quota_status` view
-- (added in 007).

BEGIN;

CREATE OR REPLACE FUNCTION _gml_update_bytes_used()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE users
            SET bytes_used = bytes_used + NEW.byte_size
            WHERE user_id = NEW.user_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE users
            SET bytes_used = GREATEST(0, bytes_used - OLD.byte_size)
            WHERE user_id = OLD.user_id;
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.user_id = OLD.user_id THEN
            UPDATE users
                SET bytes_used = bytes_used + NEW.byte_size - OLD.byte_size
                WHERE user_id = NEW.user_id;
        ELSE
            -- Memory moved between users — adjust both rows.
            UPDATE users
                SET bytes_used = GREATEST(0, bytes_used - OLD.byte_size)
                WHERE user_id = OLD.user_id;
            UPDATE users
                SET bytes_used = bytes_used + NEW.byte_size
                WHERE user_id = NEW.user_id;
        END IF;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_bytes_used ON memories;
CREATE TRIGGER memories_bytes_used
    AFTER INSERT OR UPDATE OR DELETE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION _gml_update_bytes_used();

COMMIT;
