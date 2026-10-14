-- Player-driven Guild Chat sessions, transcript, and event type.

SET @has_guild_player_message = (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'llm_chatter_events'
    AND COLUMN_NAME = 'event_type'
    AND COLUMN_TYPE LIKE '%guild_player_message%'
);

SET @sql = IF(
  @has_guild_player_message = 0,
  CONCAT(
    "ALTER TABLE `llm_chatter_events` MODIFY COLUMN ",
    "`event_type` enum(",
    "'weather_change','holiday_start','holiday_end',",
    "'creature_death_boss','creature_death_rare',",
    "'creature_death_guard','player_enters_zone',",
    "'bot_pvp_kill','bot_level_up','bot_achievement',",
    "'bot_quest_complete','world_boss_spawn','rare_spawn',",
    "'transport_arrives','day_night_transition',",
    "'enemy_player_near','bot_loot_item','bot_group_join',",
    "'bot_group_kill','bot_group_death','bot_group_loot',",
    "'bot_group_player_msg','bot_group_general_reaction',",
    "'bot_group_combat','bot_group_levelup',",
    "'bot_group_quest_complete','bot_group_achievement',",
    "'bot_group_spell_cast','bot_group_quest_objectives',",
    "'bot_group_resurrect','bot_group_zone_transition',",
    "'bot_group_dungeon_entry','bot_group_wipe',",
    "'bot_group_corpse_run','player_general_msg','minor_event',",
    "'bot_group_low_health','bot_group_oom',",
    "'bot_group_aggro_loss','bot_group_quest_accept',",
    "'bot_group_quest_accept_batch','weather_ambient',",
    "'bot_group_nearby_object','bot_group_join_batch',",
    "'bg_match_start','bg_match_end','bg_pvp_kill',",
    "'bg_flag_picked_up','bg_flag_dropped','bg_flag_captured',",
    "'bg_flag_returned','bg_node_contested','bg_node_captured',",
    "'bg_score_milestone','bg_idle_chatter','bg_player_arrival',",
    "'raid_boss_pull','raid_boss_kill','raid_boss_wipe',",
    "'raid_idle_morale','bot_group_farewell',",
    "'bot_group_subzone_change','bot_group_emote_observer',",
    "'bot_group_emote_reaction',",
    "'bot_group_screenshot_observation','proximity_say',",
    "'proximity_conversation','proximity_reply',",
    "'proximity_player_say','proximity_player_conversation',",
    "'bot_backstory_regen','bot_tone_regen',",
    "'guild_idle_chatter','guild_player_message') NOT NULL"
  ),
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

CREATE TABLE IF NOT EXISTS `llm_guild_chat_sessions` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `player_guid` INT UNSIGNED NOT NULL,
  `player_name` VARCHAR(64) NOT NULL,
  `guild_id` INT UNSIGNED NOT NULL,
  `turn_id` INT UNSIGNED NOT NULL DEFAULT 0,
  `summary` TEXT DEFAULT NULL,
  `summarized_through_id` BIGINT UNSIGNED NOT NULL DEFAULT 0,
  `started_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_activity_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_guild_session_player` (`player_guid`),
  KEY `idx_guild_session_guild` (`guild_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `llm_guild_session_history` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `session_id` BIGINT UNSIGNED NOT NULL,
  `guild_id` INT UNSIGNED NOT NULL,
  `speaker_guid` INT UNSIGNED NOT NULL DEFAULT 0,
  `speaker_name` VARCHAR(64) NOT NULL,
  `is_bot` TINYINT(1) NOT NULL DEFAULT 0,
  `source_kind` ENUM(
    'player',
    'reply',
    'ambient'
  ) NOT NULL,
  `source_event_id` INT UNSIGNED DEFAULT NULL,
  `message` TEXT NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `delivered_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_guild_history_session`
    (`session_id`, `id`),
  KEY `idx_guild_history_guild`
    (`guild_id`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
