-- Add owner_subsystem classifier to llm_chatter_messages.
--
-- Authoritative per-row owning subsystem, used by C++ delivery to
-- honor per-subsystem master toggles (e.g.
-- LLMChatter.GroupChatter.Enable) even for already-queued rows.
-- The party/raid channels are shared by group, raid-boss, and BG
-- chatter, so channel alone cannot classify a row; this column does.
--
-- Values: 'group' (party/raid group chatter — default),
-- 'raid' (PvE raid boss), 'bg' (battleground), 'general'
-- (General channel), 'proximity' (open-world say/msay),
-- 'zone' (zone-intrusion yells), 'other' (unknown channel).
--
-- Defaults to 'group' so existing pending rows and any future
-- group handler classify correctly without changes; only the
-- non-group party producers (raid_base, battlegrounds) tag
-- explicitly. See insert_chat_message() in chatter_db.py.

-- 1. Add the column only if it does not already exist (the
--    column may have been applied live ahead of this update;
--    keep the file idempotent for the DB updater).
SET @col_exists = (
    SELECT COUNT(*)
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME   = 'llm_chatter_messages'
      AND COLUMN_NAME  = 'owner_subsystem'
);

SET @sql = IF(@col_exists = 0,
    'ALTER TABLE `llm_chatter_messages` ADD COLUMN `owner_subsystem` VARCHAR(16) NOT NULL DEFAULT ''group'' AFTER `channel`',
    'SELECT 1');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 2. Backfill existing rows on UNAMBIGUOUS channels, mirroring
--    the channel->owner derivation in insert_chat_message().
--    Ambiguous party/raid rows are intentionally left as their
--    current value (default 'group') because they may be group,
--    raid-boss, or BG chatter and only the producer knows which.
--    Idempotent: re-running derives the same values.
UPDATE `llm_chatter_messages`
SET `owner_subsystem` = CASE `channel`
        WHEN 'battleground' THEN 'bg'
        WHEN 'general'      THEN 'general'
        WHEN 'say'          THEN 'proximity'
        WHEN 'msay'         THEN 'proximity'
        WHEN 'yell'         THEN 'zone'
        ELSE `owner_subsystem`
    END
WHERE `channel` IN
    ('battleground', 'general', 'say', 'msay', 'yell');
