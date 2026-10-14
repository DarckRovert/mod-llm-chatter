-- Correct hunter pet talent tab overrides installed by earlier releases.
-- Matching talenttab_dbc rows replace the client DBC rows in memory, so
-- these values must match TalentTab.dbc from WotLK 3.3.5a build 12340.

UPDATE `talenttab_dbc`
SET `PetTalentMask` = 2,
    `OrderIndex` = 0
WHERE `ID` = 409;

UPDATE `talenttab_dbc`
SET `PetTalentMask` = 1,
    `OrderIndex` = 0
WHERE `ID` = 410;

UPDATE `talenttab_dbc`
SET `PetTalentMask` = 4,
    `OrderIndex` = 0
WHERE `ID` = 411;
