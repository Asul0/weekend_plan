# agent_core/schedule_parser.py

import logging
from datetime import datetime, timedelta, time
from typing import Tuple, Optional, Dict, List
import re

logger_schedule = logging.getLogger(__name__)

DAYS_MAP_RU_TO_INT = {
    "Пн": 0,
    "Вт": 1,
    "Ср": 2,
    "Чт": 3,
    "Пт": 4,
    "Сб": 5,
    "Вс": 6,
}
DAYS_INT_TO_RU_SHORT = {v: k for k, v in DAYS_MAP_RU_TO_INT.items()}


def _parse_time_str_schedule(
    time_str: str, base_date: datetime.date, is_closing_time: bool = False
) -> Optional[datetime]:
    try:
        h, m = map(int, time_str.split(":"))
        if h == 24 and m == 0:
            if is_closing_time:
                return datetime.combine(base_date + timedelta(days=1), time.min)
            else:
                return datetime.combine(base_date, time.min)
        return datetime.combine(base_date, time(hour=h, minute=m))
    except ValueError:
        logger_schedule.warning(
            f"SCHEDULE_PARSER: Could not parse time string for schedule: {time_str} with base_date: {base_date}"
        )
        return None


def parse_schedule_and_check_open(
    schedule_str: Optional[str],
    visit_start_dt: datetime,
    desired_duration_minutes: int,
    item_type_for_schedule: str,
    min_visit_duration_minutes: int = 30,
    poi_name_for_log: str = "Unknown POI",
) -> Tuple[bool, Optional[datetime], Optional[str]]:

    logger_schedule.debug(
        f"SCHEDULE_PARSER_CHECK_OPEN (POI: '{poi_name_for_log}', Type: {item_type_for_schedule}): Args: schedule_str='{schedule_str}', visit_start_dt='{visit_start_dt}', desired_duration='{desired_duration_minutes}m', min_duration='{min_visit_duration_minutes}m'"
    )

    effective_min_duration = min_visit_duration_minutes
    if item_type_for_schedule == "park":
        effective_min_duration = 1
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Park - effective_min_duration set to {effective_min_duration}m"
        )

    if not schedule_str:
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Schedule string is missing. Assuming open for desired duration."
        )
        return (
            True,
            visit_start_dt + timedelta(minutes=desired_duration_minutes),
            "Расписание не указано",
        )

    if "круглосуточно" in schedule_str.lower():
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): POI is open 24/7."
        )
        return True, visit_start_dt + timedelta(minutes=desired_duration_minutes), None

    visit_weekday_int = visit_start_dt.weekday()
    lines = schedule_str.strip().split("\n")
    schedule_rules: List[Dict[str, any]] = []
    logger_schedule.debug(
        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Parsing {len(lines)} lines of schedule. Visit on weekday int: {visit_weekday_int} ({DAYS_INT_TO_RU_SHORT.get(visit_weekday_int, 'N/A')})"
    )

    for line_idx, line in enumerate(lines):
        line = line.strip()
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Processing line {line_idx+1}/{len(lines)}: '{line}'"
        )
        match = re.match(
            r"([А-Яа-яЁё\s,–-]+?)\s*(\d{2}:\d{2})\s*–\s*(\d{2}:\d{2})", line
        )
        if not match:
            if "ежедневно" in line.lower():
                time_match_daily = re.search(r"(\d{2}:\d{2})\s*–\s*(\d{2}:\d{2})", line)
                if time_match_daily:
                    days_range_str_daily = "Пн–Вс"
                    open_time_str_daily, close_time_str_daily = (
                        time_match_daily.groups()
                    )
                    logger_schedule.debug(
                        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): 'ежедневно' found with times {open_time_str_daily}-{close_time_str_daily}"
                    )
                    match_temp_obj_daily = re.match(
                        r"(.*?)\s*(\d{2}:\d{2})\s*–\s*(\d{2}:\d{2})",
                        f"{days_range_str_daily} {open_time_str_daily}–{close_time_str_daily}",
                    )  # Simulate full match
                    if match_temp_obj_daily:
                        match = match_temp_obj_daily
                    else:
                        logger_schedule.debug(
                            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): 'ежедневно' - internal regex construction failed for line: '{line}'. Skipping."
                        )
                        continue
                else:
                    logger_schedule.debug(
                        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): 'ежедневно' found but no time range. Skipping line: '{line}'."
                    )
                    continue
            else:
                logger_schedule.debug(
                    f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Line did not match main pattern. Skipping line: '{line}'."
                )
                continue

        days_part_str, open_time_str, close_time_str = match.groups()
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Matched line. Days part: '{days_part_str}', Open: '{open_time_str}', Close: '{close_time_str}'"
        )
        current_days_in_rule: List[int] = []
        day_tokens = re.split(r"[,\s]+", days_part_str.strip())

        for token_idx_val, token_val in enumerate(day_tokens):
            if not token_val:
                continue
            logger_schedule.debug(
                f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Processing day token {token_idx_val+1}: '{token_val}'"
            )
            if "–" in token_val or "-" in token_val:
                day_start_str_token_val, day_end_str_token_val = re.split(
                    r"[–-]", token_val
                )
                day_start_int_token_val = DAYS_MAP_RU_TO_INT.get(
                    day_start_str_token_val
                )
                day_end_int_token_val = DAYS_MAP_RU_TO_INT.get(day_end_str_token_val)
                if (
                    day_start_int_token_val is not None
                    and day_end_int_token_val is not None
                ):
                    if day_start_int_token_val <= day_end_int_token_val:
                        current_days_in_rule.extend(
                            range(day_start_int_token_val, day_end_int_token_val + 1)
                        )
                    else:
                        current_days_in_rule.extend(range(day_start_int_token_val, 7))
                        current_days_in_rule.extend(range(0, day_end_int_token_val + 1))
                    logger_schedule.debug(
                        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Parsed day range '{token_val}' to ints: {current_days_in_rule[- (day_end_int_token_val - day_start_int_token_val + 1 if day_start_int_token_val <= day_end_int_token_val else 7 - day_start_int_token_val + day_end_int_token_val + 1):]}"
                    )
            else:
                day_int_token_val = DAYS_MAP_RU_TO_INT.get(token_val)
                if day_int_token_val is not None:
                    current_days_in_rule.append(day_int_token_val)
                logger_schedule.debug(
                    f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Parsed single day token '{token_val}' to int: {day_int_token_val}"
                )

        if not current_days_in_rule:
            logger_schedule.debug(
                f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Could not parse any days from days_part_str: '{days_part_str}'. Skipping rule."
            )
            continue
        schedule_rules.append(
            {
                "days_int": list(set(current_days_in_rule)),
                "open_str": open_time_str,
                "close_str": close_time_str,
                "original_line": line,
            }
        )

    if not schedule_rules:
        logger_schedule.warning(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Could not parse any rule from schedule: '{schedule_str}'. Assuming open."
        )
        return (
            True,
            visit_start_dt + timedelta(minutes=desired_duration_minutes),
            "Не удалось точно определить часы работы",
        )

    applicable_rule_found = None
    for rule_item_val in schedule_rules:
        if visit_weekday_int in rule_item_val["days_int"]:
            applicable_rule_found = rule_item_val
            logger_schedule.debug(
                f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Found applicable rule for visit day {visit_weekday_int} ({DAYS_INT_TO_RU_SHORT.get(visit_weekday_int)}): '{rule_item_val['original_line']}'"
            )
            break

    if not applicable_rule_found:
        logger_schedule.warning(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): No applicable schedule rule found for visit day {visit_weekday_int} ({DAYS_INT_TO_RU_SHORT.get(visit_weekday_int)}) in schedule:\n{schedule_str}"
        )
        return (
            False,
            None,
            f"Нет информации о работе в {DAYS_INT_TO_RU_SHORT.get(visit_weekday_int, 'этот день')}",
        )

    rule_open_dt_parsed = _parse_time_str_schedule(
        applicable_rule_found["open_str"], visit_start_dt.date(), is_closing_time=False
    )
    rule_close_dt_parsed = _parse_time_str_schedule(
        applicable_rule_found["close_str"], visit_start_dt.date(), is_closing_time=True
    )

    if not rule_open_dt_parsed or not rule_close_dt_parsed:
        logger_schedule.warning(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Could not parse open/close times from rule: {applicable_rule_found}"
        )
        return False, None, "Ошибка разбора времени работы из правила"

    rule_open_dt_effective = rule_open_dt_parsed
    rule_close_dt_effective = rule_close_dt_parsed

    if rule_open_dt_parsed.time() > rule_close_dt_parsed.time() and not (
        rule_close_dt_parsed.time() == time.min and rule_close_dt_parsed.hour == 0
    ):
        rule_close_dt_effective = rule_close_dt_parsed + timedelta(days=1)
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Close time '{applicable_rule_found['close_str']}' for rule '{applicable_rule_found['original_line']}' is on the next day. Effective close: {rule_close_dt_effective}"
        )

    if visit_start_dt < rule_open_dt_effective:
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Visit time {visit_start_dt} is before today's rule open time {rule_open_dt_effective}. Checking yesterday's rule if applicable."
        )
        yesterday_visit_weekday_int_val = (visit_weekday_int - 1 + 7) % 7
        yesterday_rule_val = None
        for rule_item_yst_val in schedule_rules:
            if yesterday_visit_weekday_int_val in rule_item_yst_val["days_int"]:
                yesterday_rule_val = rule_item_yst_val
                logger_schedule.debug(
                    f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Found yesterday's rule: '{yesterday_rule_val['original_line']}'"
                )
                break
        if yesterday_rule_val:
            yst_open_dt_val = _parse_time_str_schedule(
                yesterday_rule_val["open_str"],
                visit_start_dt.date() - timedelta(days=1),
                is_closing_time=False,
            )
            yst_close_dt_val = _parse_time_str_schedule(
                yesterday_rule_val["close_str"],
                visit_start_dt.date() - timedelta(days=1),
                is_closing_time=True,
            )
            if yst_open_dt_val and yst_close_dt_val:
                yst_close_dt_effective_val = yst_close_dt_val
                if yst_open_dt_val.time() > yst_close_dt_val.time() and not (
                    yst_close_dt_val.time() == time.min and yst_close_dt_val.hour == 0
                ):
                    yst_close_dt_effective_val = yst_close_dt_val + timedelta(days=1)
                logger_schedule.debug(
                    f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Yesterday's effective rule: Open {yst_open_dt_val}, Close {yst_close_dt_effective_val}"
                )
                if yst_open_dt_val <= visit_start_dt < yst_close_dt_effective_val:
                    logger_schedule.debug(
                        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Visit time {visit_start_dt} falls into yesterday's extended hours. Using yesterday's rule."
                    )
                    rule_open_dt_effective = yst_open_dt_val
                    rule_close_dt_effective = yst_close_dt_effective_val
                else:
                    logger_schedule.debug(
                        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Visit time {visit_start_dt} does not fall into yesterday's extended hours. Sticking to today's rule logic."
                    )
            else:
                logger_schedule.debug(
                    f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Could not parse times for yesterday's rule."
                )
        else:
            logger_schedule.debug(
                f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): No rule found for yesterday to check for overnight tail."
            )

    logger_schedule.debug(
        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Final Effective Rule: Open {rule_open_dt_effective}, Close {rule_close_dt_effective} (for visit at {visit_start_dt})"
    )

    if not (rule_open_dt_effective <= visit_start_dt < rule_close_dt_effective):
        logger_schedule.info(
            f"SCHEDULE_PARSER_CLOSED ({poi_name_for_log}): POI determined CLOSED at visit_start_dt ({visit_start_dt}). Effective Rule Open: {rule_open_dt_effective}, Effective Rule Close: {rule_close_dt_effective}."
        )
        return False, None, f"Заведение закрыто в {visit_start_dt.strftime('%H:%M')}"

    max_possible_visit_end_dt = visit_start_dt + timedelta(
        minutes=desired_duration_minutes
    )
    logger_schedule.debug(
        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Initial max_possible_visit_end_dt (desired_duration): {max_possible_visit_end_dt}"
    )

    if max_possible_visit_end_dt > rule_close_dt_effective:
        logger_schedule.debug(
            f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Desired end_dt {max_possible_visit_end_dt} is after POI close time {rule_close_dt_effective}. Adjusting to close time."
        )
        max_possible_visit_end_dt = rule_close_dt_effective

    actual_visit_duration_td = max_possible_visit_end_dt - visit_start_dt
    logger_schedule.debug(
        f"SCHEDULE_PARSER_CHECK_OPEN ({poi_name_for_log}): Actual visit duration based on close time: {actual_visit_duration_td.total_seconds()/60:.0f}m. Effective min duration needed: {effective_min_duration}m"
    )

    if actual_visit_duration_td < timedelta(minutes=effective_min_duration):
        logger_schedule.info(
            f"SCHEDULE_PARSER_TOO_SHORT ({poi_name_for_log}): Adjusted visit duration {actual_visit_duration_td.total_seconds()/60:.0f}m is less than effective_min_duration {effective_min_duration}m."
        )
        return (
            False,
            None,
            f"Остается слишком мало времени до закрытия (менее {effective_min_duration} мин)",
        )

    logger_schedule.info(
        f"SCHEDULE_PARSER_OPEN ({poi_name_for_log}): POI determined OPEN. Visit from {visit_start_dt} to {max_possible_visit_end_dt}. Duration: {actual_visit_duration_td.total_seconds()/60:.0f}m."
    )
    return True, max_possible_visit_end_dt, None
