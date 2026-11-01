-- PROJECT JAINA: Migration to prevent MySQL Errno 3140 (Invalid JSON text)
-- Converts extra_data from strict native JSON to LONGTEXT.
-- Preserves full JSON compatibility with MySQL JSON_EXTRACT / JSON_UNQUOTE functions
-- and python json.loads() while preventing insertion drops due to escaping quirks.
ALTER TABLE `llm_chatter_events` MODIFY COLUMN `extra_data` LONGTEXT DEFAULT NULL;
